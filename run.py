import torch
import csv
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, expansion=0):
        super(block,self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(nn.Linear(in_features=in_channels,out_features=out_channels+expansion,bias=True))
        for i in range(depth-2):
            self.layers.append(nn.Linear(in_features=out_channels+expansion, out_features=out_channels+expansion,bias=True))
        self.layers.append(nn.Linear(in_features=out_channels+expansion, out_features=out_channels, bias=True))
        
    def forward(self,x):
        identity=x
        for layer in self.layers:
            x=layer(x)
        x+=identity
        return x
    
class aicure(nn.Module):
    def __init__(self, block, block_config):
        super(aicure,self).__init__()
        self.block_layers=nn.ModuleList()
        for i in block_config:
#             print(i)
            self.block_layers.append(block(in_channels=i[0],out_channels=i[1],depth=i[2],expansion=i[3]))
#         print(self.block_layers)
        
        self.fc=nn.Linear(in_features=block_config[-1][1], out_features=1, bias=True)
        self.blocks=nn.Sequential(*self.block_layers)
#         print(self.blocks)
        
    def forward(self,x):
        x=self.blocks(x)        
        x=self.fc(x)
        return x
    
def model_aicure(block_config):
    return aicure(block, block_config)

cond={
    "interruption":1,
    "time pressure":2,
    "no stress":3
}
    

block_config=[
    [34,34,5,2],
    [34,34,5,2],
    [34,34,5,2]

]

model=model_aicure(block_config)
modelpth=torch.load("model.pth")
model.load_state_dict(modelpth['model'])

infile="sample_test_data.csv"
# outfile="sample_output_generated.csv"
inf=open(infile,"r")
# outf=open(outfile,"r")
indata=csv.reader(inf)
# outdata=csv.reader(outf)
ind=[]
for i in indata:
    ind.append(i)
    
    
uuid=[]
for i in range(len(ind)):
    ind[i].pop(16)
    uuid.append(ind[i].pop(0))
#     print(len(ind[i]))

uuid.pop(0)
fields_in=ind.pop(0)
fields_in.pop(15)

for i in range(len(ind)):
    ind[i][15]=cond[ind[i][15]]
    
for i in range(len(ind)):
    for j in range(len(ind[i])):
        ind[i][j]=float(ind[i][j])
    
for i in range(len(ind)):
    ind[i]=torch.tensor(ind[i])
    
final_results=[]
for i in range(len(ind)):
    final_results.append([uuid[i],model(ind[i]).item()])
# print(final_results)
    
# print(final_results)

# outd=[]
# for i in outdata:
#     outd.append(i)
# print(outd)


# outd.pop(0)

# for i in range(len(outd)):
#         outd[i][1]=float(outd[i][1])


# for i in range(len(outd)):
    # print(outd[i][0],abs(outd[i][1]-final_results[i][1]))
    
outfile="results.csv"

with open(outfile,'w') as f:
    csv_writer = csv.writer(f)
    
    csv_writer.writerow(['uuid',"HR"])
    
    csv_writer.writerows(final_results)

    