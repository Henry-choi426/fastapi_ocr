from easyocr import *
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
# Attn ver
# from dataset import AlignCollate, RawDataset2
# from utils import AttnLabelConverter
# from model.easyOCR import model_test

# class opt():
#     imgH = 32
#     imgW = 100
#     num_fiducial = 20
#     input_channel = 1
#     output_channel = 512
#     hidden_size = 256
#     num_class = 38
#     batch_max_length = 25
#     image_folder ='./demo/'
#     workers = 40
#     batch_size = 16
#     rgb = False
#     Transformation = 'TPS'
#     FeatureExtraction = 'ResNet'
#     SequenceModeling = 'BiLSTM'
#     Prediction = 'Attn'
#     character = '0123456789abcdefghijklmnopqrstuvwxyz'
    
class ApiConfig():
# Attn Ver
#     AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     eng_model = model_test.Model(opt)
    
#     converter = AttnLabelConverter(opt.character)
#     opt.num_class = len(converter.character)
#     eng_model = torch.nn.DataParallel(eng_model).to(device)
#     eng_model.load_state_dict(torch.load('./model/easyOCR/Attn.pth', map_location=device))


    
    names = ['부산광역시','충청북도','충청남도','대구광역시','대전광역시','강원도','광주광역시','경상북도','경상남도','경기도','인천광역시','제주특별자치도','전라북도','전라남도','세종특별자치시','서울특별시','울산광역시']
    add_dict = dict()
    for named in names:
        with open("./unique_address/"+named, "rb" ) as file:
            loaded_data = pickle.load(file)
            add_dict[named] = loaded_data
    

    qualification = {'A-1': '외 교',
                     'A-2': '공 무',
                     'A-3': '협 정',
                     'B-1': '사증면제',
                     'B-2': '관광·통과',
                     'C-1': '일시취재',
                     'C-3': '단기방문',
                     'C-4': '단기취업',
                     'D-1': '문화예술',
                     'D-2': '유 학',
                     'D-3': '기술연수',
                     'D-4': '일반연수',
                     'D-5': '취재',
                     'D-6': '종교',
                     'D-7': '주재',
                     'D-8': '기업투자',
                     'D-9': '무역경영',
                     'D-10': '구직',
                     'E-1': '교수',
                     'E-2': '회화지도',
                     'E-3': '연구',
                     'E-4': '기술지도',
                     'E-5': '전문직업',
                     'E-6': '예술흥행',
                     'E-7': '특정활동',
                     'E-9': '비전문취업',
                     'E-10': '선원취업',
                     'F-1': '방문동거',
                     'F-2': '거주',
                     'F-3': '동반',
                     'F-4': '재외동포',
                     'F-5': '영주',
                     'F-6': '결혼이민',
                     'G-1': '불법체류',
                     'H-2': '방문취업'}