import argparse
from torch import nn
from torch.tensor import OrderedDict
import torch
import numpy as np
import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from sas7bdat import SAS7BDAT

from collagen.core import Module
import solt.data as sld
from collagen.data.utils import ApplyTransform, Normalize, Compose
import solt.core as slc
import solt.transforms as slt


oai_meta_csv_filename = "oai_meta.csv"
most_meta_csv_filename = "most_meta.csv"
oai_participants_csv_filename = "oai_participants.csv"
most_participants_csv_filename = "most_participants.csv"
oai_most_meta_csv_filename = "oai_most_meta.csv"
oai_most_all_csv_filename = "oai_most_all.csv"


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])


def build_img_klg_meta_oai(oai_src_dir):
    #     visits = ['00', '12', '24', '36', '72', '96']
    dataset_name = "oai"
    visits = ['00']
    exam_codes = ['00', '01', '03', '05', '08', '10']
    rows = []
    sides = [None, 'R', 'L']
    for i, visit in enumerate(visits):
        print(f'==> Reading OAI {visit} visit')
        meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                              'Semi-Quant Scoring_SAS',
                                              f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
        # Dropping the data from multiple projects
        meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
        meta.fillna(-1, inplace=True)
        for c in meta.columns:
            meta[c.upper()] = meta[c]

        meta['KL'] = meta[f'V{exam_codes[i]}XRKL']
        for index, row in tqdm(meta.iterrows(), total=len(meta.index), desc="Loading OAI meta"):
            _s = int(row['SIDE'])
            if sides[_s] is None:
                continue
            rows.append({'ID': row['ID'], 'Side': sides[_s], 'KL': row['KL'], 'dataset': dataset_name})

    return pd.DataFrame(rows, index=None)


def build_img_klg_meta_most(most_src_dir):
    dataset_name = "most"
    data = read_sas7bdata_pd(os.path.join(most_src_dir, 'mostv01235xray.sas7bdat')).fillna(-1)
    data.set_index('MOSTID', inplace=True)
    rows = []
    # Assumption: Only use visit 0 (baseline)
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}

    enrolled = {}
    for visit in [0]:
        print(f'==> Reading MOST {visit} visit')
        ds = read_sas7bdata_pd(files_dict[f'mostv{visit}enroll.sas7bdat'])
        ds = ds[ds['V{}PA'.format(visit)] == 1]  # Filtering out the cases when X-rays wern't taken
        id_set = set(ds.MOSTID.values.tolist())
        enrolled[visit] = id_set

    rows = []
    for ID in tqdm(enrolled[0], total=len(enrolled[0]), desc="Loading MOST meta"):
        subj = data.loc[ID]
        KL_bl_l = subj['V{0}X{1}{2}'.format(0, 'L', 'KL')]
        KL_bl_r = subj['V{0}X{1}{2}'.format(0, 'R', 'KL')]
        rows.append({'ID': ID, 'Side': 'L', 'KL': KL_bl_l, 'dataset': dataset_name})
        rows.append({'ID': ID, 'Side': 'R', 'KL': KL_bl_r, 'dataset': dataset_name})

    return pd.DataFrame(rows, columns=['ID', 'Side', 'KL', 'dataset'])


def build_clinical_oai(oai_src_dir):
    data_enrollees = read_sas7bdata_pd(os.path.join(oai_src_dir, 'enrollees.sas7bdat'))
    data_clinical = read_sas7bdata_pd(os.path.join(oai_src_dir, 'allclinical00.sas7bdat'))

    clinical_data_oai = data_clinical.merge(data_enrollees, on='ID')

    # Age, Sex, BMI
    clinical_data_oai['SEX'] = 2 - clinical_data_oai['P02SEX']
    clinical_data_oai['AGE'] = clinical_data_oai['V00AGE']
    clinical_data_oai['BMI'] = clinical_data_oai['P01BMI']

    clinical_data_oai_left = clinical_data_oai.copy()
    clinical_data_oai_right = clinical_data_oai.copy()

    # Making side-wise metadata
    clinical_data_oai_left['Side'] = 'L'
    clinical_data_oai_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_oai_left['INJ'] = clinical_data_oai_left['P01INJL']
    clinical_data_oai_right['INJ'] = clinical_data_oai_right['P01INJR']

    # Surgery (ever had)
    clinical_data_oai_left['SURG'] = clinical_data_oai_left['P01KSURGL']
    clinical_data_oai_right['SURG'] = clinical_data_oai_right['P01KSURGR']

    # Total WOMAC score
    clinical_data_oai_left['WOMAC'] = clinical_data_oai_left['V00WOMTSL']
    clinical_data_oai_right['WOMAC'] = clinical_data_oai_right['V00WOMTSR']

    clinical_data_oai = pd.concat((clinical_data_oai_left, clinical_data_oai_right))
    clinical_data_oai.ID = clinical_data_oai.ID.values.astype(str)
    return clinical_data_oai[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC']]


def build_clinical_most(most_src_dir):
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    clinical_data_most = read_sas7bdata_pd(files_dict['mostv0enroll.sas7bdat'])
    clinical_data_most['ID'] = clinical_data_most.MOSTID
    clinical_data_most['BMI'] = clinical_data_most['V0BMI']

    clinical_data_most_left = clinical_data_most.copy()
    clinical_data_most_right = clinical_data_most.copy()

    # Making side-wise metadata
    clinical_data_most_left['Side'] = 'L'
    clinical_data_most_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_most_left['INJ'] = clinical_data_most_left['V0LAL']
    clinical_data_most_right['INJ'] = clinical_data_most_right['V0LAR']

    # Surgery (ever had)
    clinical_data_most_left['SURG'] = clinical_data_most_left['V0SURGL']
    clinical_data_most_right['SURG'] = clinical_data_most_right['V0SURGR']

    # Total WOMAC score
    clinical_data_most_left['WOMAC'] = clinical_data_most_left['V0WOTOTL']
    clinical_data_most_right['WOMAC'] = clinical_data_most_right['V0WOTOTR']

    clinical_data_most = pd.concat((clinical_data_most_left, clinical_data_most_right))

    return clinical_data_most[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC']]


def load_oai_most_datasets(root, save_dir, force_reload=False):
    oai_meta_fullname = os.path.join(save_dir, oai_meta_csv_filename)
    oai_participants_fullname = os.path.join(save_dir, oai_participants_csv_filename)
    most_meta_fullname = os.path.join(save_dir, most_meta_csv_filename)
    most_participants_fullname = os.path.join(save_dir, oai_participants_csv_filename)
    oai_most_meta_fullname = os.path.join(save_dir, oai_most_meta_csv_filename)
    oai_most_all_fullname = os.path.join(save_dir, oai_most_all_csv_filename)

    requires_update = False

    if os.path.isfile(oai_meta_fullname) and not force_reload:
        oai_meta = pd.read_csv(oai_meta_fullname)
    else:
        oai_meta = build_img_klg_meta_oai(os.path.join(root, 'X-Ray_Image_Assessments_SAS/'))
        oai_meta.to_csv(oai_meta_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(oai_participants_fullname) and not force_reload:
        oai_ppl = pd.read_csv(oai_participants_fullname)
    else:
        oai_ppl = build_clinical_oai(os.path.join(root, 'X-Ray_Image_Assessments_SAS/'))
        oai_ppl.to_csv(oai_participants_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(most_meta_fullname) and not force_reload:
        most_meta = pd.read_csv(most_meta_fullname)
    else:
        most_meta = build_img_klg_meta_most(os.path.join(root, 'most_meta/'))
        most_meta.to_csv(most_meta_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(most_participants_fullname) and not force_reload:
        most_ppl = pd.read_csv(most_participants_fullname)
    else:
        most_ppl = build_clinical_most(os.path.join(root, 'most_meta/'))
        most_ppl.to_csv(most_participants_fullname, index=None, sep='|')
        requires_update = True

    master_dict = {"oai": dict(), "most": dict()}
    master_dict["oai"]["meta"] = oai_meta
    master_dict["oai"]["ppl"] = oai_ppl
    master_dict["most"]["meta"] = most_meta
    master_dict["most"]["ppl"] = most_ppl

    master_dict["oai"]["n_dup"] = dict()
    master_dict["most"]["n_dup"] = dict()
    master_dict["oai"]["n_dup"]["meta"] = len(oai_meta[oai_meta.duplicated(keep=False)].index)
    master_dict["oai"]["n_dup"]["ppl"] = len(oai_ppl[oai_ppl.duplicated(keep=False)].index)
    master_dict["most"]["n_dup"]["meta"] = len(most_meta[most_meta.duplicated(keep=False)].index)
    master_dict["most"]["n_dup"]["ppl"] = len(most_ppl[most_ppl.duplicated(keep=False)].index)

    for ds in ["oai", "most"]:
        for s in ["meta", "ppl"]:
            if master_dict[ds]["n_dup"][s] > 0:
                print(master_dict[ds][s])
                raise ValueError("There are {} duplicated rows in {} {} dataframe".format(master_dict[ds]["n_dup"][s], ds.upper(),
                                                                               s.upper()))

    master_dict["oai_most"] = dict()
    master_dict["oai_most"]["meta"] = pd.concat([master_dict["oai"]["meta"],
                                                 master_dict["most"]["meta"]], ignore_index=True)

    master_dict["oai"]["all"] = pd.merge(master_dict["oai"]["meta"],
                                         master_dict["oai"]["ppl"], how="left",
                                         left_on=["ID", "Side"], right_on=["ID", "Side"]).fillna(-1)
    master_dict["most"]["all"] = pd.merge(master_dict["most"]["meta"],
                                          master_dict["most"]["ppl"], how="left",
                                          left_on=["ID", "Side"], right_on=["ID", "Side"]).fillna(-1)

    master_dict["oai_most"]["all"] = pd.concat([master_dict["oai"]["all"],
                                                master_dict["most"]["all"]], ignore_index=True)

    if requires_update:
        master_dict["oai_most"]["meta"].to_csv(oai_most_meta_fullname, index=None, sep='|')
        master_dict["oai_most"]["all"].to_csv(oai_most_all_fullname, index=None, sep='|')

    return master_dict


def crop_img(img, crop_side_ratio=0.55):
    # Assumption: Input image is square
    sz = round(img.shape[0]*crop_side_ratio)
    center = (img.shape[0]//2, img.shape[1]//2)

    tl1 = (center[0] - sz//2, 0)
    br1 = (tl1[0] + sz, tl1[1] + sz)

    br2 = (br1[0], img.shape[1] - 1)
    tl2 = (br2[0] - sz, br2[1] - sz)

    img1 = img[tl1[0]:br1[0], tl1[1]:br1[1]]
    img2 = img[tl2[0]:br2[0], tl2[1]:br2[1]]

    return img1, img2


def load_meta_with_imgs(df, img_dir, saved_patch_dir, force_rewrite=False):
    if not os.path.exists(saved_patch_dir):
        os.mkdir(saved_patch_dir)

    fullnames = []
    for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing images"):
        fname = row["ID"] + "_00_" + row["Side"] + ".png"
        img_fullname = os.path.join(img_dir, fname)
        basename = os.path.splitext(fname)[0]
        img1_fullname = os.path.join(saved_patch_dir, "{}_patch1.png".format(basename))
        img2_fullname = os.path.join(saved_patch_dir, "{}_patch2.png".format(basename))
        fullnames_dict = row.to_dict()
        fullnames_dict['Filename'] = None
        fullnames_dict['Patch1_name'] = None
        fullnames_dict['Patch2_name'] = None

        fullnames_dict['ID'] = row['ID']
        fullnames_dict['Side'] = row['Side']
        if os.path.exists(img_fullname):
            if not os.path.isfile(img1_fullname) or not os.path.isfile(img2_fullname) or force_rewrite:
                img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)

                # Flip if right knee
                if row['Side'] == 'R':
                    img = img[:,::-1]

                img1, img2 = crop_img(img, crop_side_ratio=0.55)
                cv2.imwrite(img1_fullname, img1)
                cv2.imwrite(img2_fullname, img2)

            fullnames_dict['Filename'] = fname
            fullnames_dict['Patch1_name'] = os.path.basename(img1_fullname)
            fullnames_dict['Patch2_name'] = os.path.basename(img2_fullname)
            fullnames.append(fullnames_dict)

    return pd.DataFrame(fullnames, index=None)


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img/255.0, np.float32(target)


def init_mnist_transforms():
    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.ResizeTransform(resize_to=(64, 64), interpolation='bilinear'),
            slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
            slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
            slt.RandomRotate(rotation_range=(-10, 10), p=0.5),
            # slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
            slt.PadTransform(pad_to=70),
            slt.CropTransform(crop_size=64, crop_mode='r')
        ]),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])

    test_trf = Compose([
        wrap2solt,
        slt.ResizeTransform(resize_to=(64, 64), interpolation='bilinear'),
        # slt.PadTransform(pad_to=64),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,))),

    ])

    return train_trf, test_trf


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(Module):
    def __init__(self, nc=1, ndf=64, n_cls=10, ngpu=1, drop_rate=0.35):
        super(Discriminator, self).__init__()
        # input is (nc) x 32 x 32
        self.__ngpu = ngpu
        self.__drop_rate = drop_rate

        self.dropout = nn.Dropout(p=self.__drop_rate)
        # input is (nc) x 64 x 64
        self._layer1 = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ndf * 2),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*2) x 16 x 16

        self._layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ndf * 4),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*4) x 8 x 8

        self._layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ndf * 8),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*4) x 4 x 4

        # self._layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #                              nn.Sigmoid())  # state size. 1x1x1

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    # ("dropout1", self.dropout),
                                                    ("conv_block2", self._layer2),
                                                    # ("dropout2", self.dropout),
                                                    ("conv_block3", self._layer3),
                                                    # ("dropout3", self.dropout),
                                                    ("conv_block4", self._layer4),
                                                    # ("dropout3", self.dropout),
                                                    # ("conv_final", self._layer5)
                                                    ]))

        self.valid = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                                   nn.Sigmoid())  # state size. 1x1x1

        self.classify = nn.Sequential(nn.Conv2d(ndf * 8, n_cls, 4, 1, 0, bias=False),
                                      nn.Softmax(dim=1))  # state size. n_clsx1x1

        self.apply(weights_init)

    def forward(self, x):
        o3 = self.main_flow(x)
        validator = self.valid(o3).squeeze(-1).squeeze(-1)
        classifier = self.classify(o3).squeeze(-1).squeeze(-1)
        return torch.cat((classifier, validator), dim=-1)


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64, ngpu=1):
        super(Generator, self).__init__()
        self.__ngpu = ngpu

        self._layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                     nn.BatchNorm2d(ngf * 8),
                                     nn.ReLU(True))  # state size. (ngf*8) x 4 x 4

        self._layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf * 4),
                                     nn.ReLU(True))  # state size. (ngf*2) x 8 x 8

        self._layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf * 2),
                                     nn.ReLU(True))  # state size. (ngf*2) x 16 x 16

        self._layer4 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf),
                                     nn.ReLU(True))  # state size. (ngf) x 32 x 32

        self._layer5 = nn.Sequential(nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
                                     nn.Tanh())  # state size. (nc) x 64 x 64

        # self._layer6 = nn.Sequential(nn.Conv2d(ngf // 2, 1, 3, 1, 1, bias=False),
        #                              nn.Sigmoid())  # state size. (ngf) x 64 x 64

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    ("conv_block2", self._layer2),
                                                    ("conv_block3", self._layer3),
                                                    ("conv_block4", self._layer4),
                                                    ("conv_block5", self._layer5),
                                                    # ("conv_final", self._layer6)
                                                    ]))

        self.main_flow.apply(weights_init)

    def forward(self, x):
        if len(x.size()) != 2:
            raise ValueError("Input must have 2 dim but found {}".format(x.shape))
        x = x.view(x.size(0), x.size(1), 1, 1)

        if x.is_cuda and self.__ngpu > 1:
            output = nn.parallel.data_parallel(self.main_flow, x, range(self.__ngpu))
        else:
            output = self.main_flow(x)

        return output


def parse_item_mnist_gan(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'data': img, 'target': np.float32(1.0), 'class': target}


def parse_item_mnist_ssgan(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    # ext_y = np.zeros(2, dtype=np.int64)
    # ext_y[0] = 1
    # ext_y[1] = target
    ext_y = np.zeros(11, dtype=np.float32)
    ext_y[-1] = 1.0
    ext_y[int(round(target))] = 1.0
    return {'data': img, 'target': ext_y, 'class': target, 'valid': ext_y[-1]}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--d_wd', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_wd', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--beta1', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=10, help='Num of classes')
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--grid_shape', type=tuple, default=(24, 24), help='Shape of grid of generated images')
    parser.add_argument('--ngpu', type=int, default=1, help='Num of GPUs')

    parser.add_argument('--oai_meta', default='./data/X-Ray_Image_Assessments_SAS')
    parser.add_argument('--most_meta', default='./data/most_meta')
    parser.add_argument('--img_dir', default='./data/MOST_OAI_00_0_2')
    parser.add_argument('--save_meta', default='./Metadata/')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args




