from utility import * 

data_dir = "flowers"


def run(arch,learning_rate,hidden_units,num_epochs,save_dir,device):   
    data_transforms = transformations()
    image_datasets = load_image_datasets(data_transforms)
    data_loaders = load_data_loaders(image_datasets)
    data_size = load_data_size(image_datasets)
    num_classes = len(image_datasets['train'].classes)
    model = prepare_pretrained(arch,hidden_units,num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    lr_scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    
    model = train(model ,optimizer ,criterion,lr_scheduler,num_epochs,data_loaders,data_size,device)
    model.class_to_idx = image_datasets['train'].class_to_idx
    test(model,criterion,data_loaders,data_size,device)
    save_checkpoint(save_dir ,model)
    
    
    

def set_device(enable_gpu = False) :
    if enable_gpu and torch.cuda.is_available() :
        device = torch.device("cuda:0")
    else :
        device = torch.device("cpu")
        
    return device    

 

def train(model ,optimizer ,criterion,lr_scheduler,num_epochs,data_loaders,data_size,device):
    begin = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #for epoch in tnrange(num_epochs ,desc="Epoch") :
    for epoch in tqdm(range(num_epochs) ,desc = "Epochs") :   
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 20)

        for phase in ['train', 'valid']:
            if phase == 'train': model.train()  
            else: model.eval()   
            avg_loss = 0.0
            avg_correct = 0

            for index, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = F.log_softmax(outputs,dim = 1).max(dim=1)[1]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                avg_loss += loss.item() * inputs.size(0)
                avg_correct += torch.sum(preds == labels.data)

            if phase == 'train': lr_scheduler.step()
            epoch_loss = avg_loss / data_size[phase]
            epoch_acc = avg_correct.double() / data_size[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - begin
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model)
    return model

def test(model,criterion,data_loaders,data_size,device):
    model.eval()   
    avg_loss = 0.0
    avg_correct = 0
    for index, (inputs, labels) in enumerate(data_loaders['test']):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = F.log_softmax(outputs,dim = 1).max(dim=1)[1]

            avg_loss += loss.item() * inputs.size(0)
            avg_correct += torch.sum(preds == labels.data)

    total_loss = avg_loss / data_size['test']
    total_acc = avg_correct.double() / data_size['test']
    print(f'test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')
    
def transformations():
    resize = 224
    data_transforms = {
        'train' :
                transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'valid' :
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'test' :
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    }
    return data_transforms
    
def load_image_datasets(data_transforms):
    image_datasets = {
        x: datasets.ImageFolder( root = os.path.join(data_dir, x),transform=data_transforms[x]) for x in ['train', 'valid', 'test']
    }
    return image_datasets
    
def load_data_loaders(image_datasets,batch_size = 64):
    data_loaders = {
        'train' : DataLoader (image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid' : DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False),
        'test' :  DataLoader(image_datasets['test'],  batch_size=batch_size, shuffle=False)
    }
    return data_loaders
    
def load_data_size(image_datasets) :
    data_size = {x : len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    return data_size
    
    
def prepare_pretrained(arch , hidden_units,num_classes):
    if hasattr(models, arch) :
        model = getattr(models,arch)(pretrained = True)
    else :
        model = models.vgg_19_bn(pretrained = True)
        
        
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_ftrs  = list(model.classifier.children())[0].in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units,bias = True)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, hidden_units ,bias = True)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_units, num_classes ,bias = True))
                          ]))
    
    model.classifier = classifier 
    return model
    
        
        
    
def save_checkpoint(file_name ,model):      
    torch.save({
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
                ,file_name)
    
    
  

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,help="path to image directories")
    ap.add_argument("-s", "--save_dir", required=False,help="Set directory to save checkpoints")
    ap.add_argument("-a", "--arch", required=False,help="Choose architecture")  
    ap.add_argument("-lr", "--learning_rate", required=False,help="Set learning_rate" ,type=float)
    ap.add_argument("-hd", "--hidden_units", required=False,help="Set hidden_units" ,type=int)
    ap.add_argument("-ep", "--epochs", required=False,help="Set epochs",type=int)
    ap.add_argument("-gp", "--gpu", required=False,help="Set device")
    args = vars(ap.parse_args())
   
    
    data_dir = args['input']
    
    save_dir = 'save_check.pth'
    if args['save_dir'] is not None:
        save_dir = args['save_dir']
        
        
    arch = "vgg19_bn"
    if args['arch'] is not None:
        arch = args['arch']
    
    
    learning_rate = 0.001
    if args['learning_rate'] is not None:
        learning_rate = args['learning_rate']
        
    hidden_units = 4069
    if args['hidden_units'] is not None:
        hidden_units = args['hidden_units'] 
    
        
    epochs = 8
    if args['epochs'] is not None:
        epochs = args['epochs']  

    device = "cpu"
    if args['gpu'] is not None:
        device = set_device(enable_gpu = True)
        print("Setting device" , device)
        
        
    run(arch,learning_rate,hidden_units,epochs,save_dir,device)
        
        
        
    