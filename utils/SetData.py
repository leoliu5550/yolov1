import torch 
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
import os 

class ChessDataset(Dataset):
    def __init__(self,path,levels ='train',S = 7, B = 2, C = 12,transform =None):
        self.path = os.path.join(path,levels)
        self.image_path = os.path.join(self.path,'images')
        self.label_path = os.path.join(self.path,'labels')
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.imageFilelist = os.listdir(self.image_path)
        self.FileKeylist = [i[:-4] for i in self.imageFilelist]
        self.KeyDict = {}
        for ke in self.FileKeylist:
            temp = {
                '__image':os.path.join(self.image_path,ke)+'.jpg',
                '__label':os.path.join(self.label_path,ke)+'.txt'
            }
            self.KeyDict[ke] = temp
    def __len__(self):
        return len(self.FileKeylist)
    
    def __getitem__(self,index):
        keys = self.FileKeylist[index]
        print(keys)
        boxes = []
        with open(self.KeyDict[keys]['__label'],'r') as file:
            for label in file.readlines():
                label = label.strip().split(' ')
                cls_label , x,y,w,h = label
                boxes.append([int(cls_label) ,float(x),float(y),float(w),float(h)])
        image = read_image(self.KeyDict[keys]['__image'])/255

        if self.transform:
            boxes = torch.tensor(boxes)
            image,boxes = self.transform(image,boxes)
            boxes = boxes.tolist()
        label_matrix = torch.zeros(
            self.S,
            self.S,
            self.C + 5 # o,x,y,w,h
            )
        for box in boxes:
            cls_label , x,y,w,h = box
            i,j = int(self.S * y),int(self.S * x)
            x_cell,y_cell = self.S * x -j , self.S * y -i
            w_cell,h_cell = self.S * w, self.S * h
            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            # [0:self.c] = classP , [self.C] = Pexist, [self.C+1:] = box
            if label_matrix[i,j,self.C] == 0:
                print(i,j)
                label_matrix[i,j,self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i,j,self.C+1:] = box_coordinates
                label_matrix[i,j,cls_label] = 1


        return image,label_matrix


        
# dataset = ChessDataset(path = 'DATA')
# # print(dataset.KeyDict)
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# epochs =1
# N=0
# for epoch in range(epochs):

#     for i, (features, targets) in enumerate(dataloader):
#         N+=1
#         print(i, (features, targets))
#     print(N)
#     break