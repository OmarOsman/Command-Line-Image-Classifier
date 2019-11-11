
from utility import * 



def run(image_path,checkpoint,top_k,cat_to_name,device):   
    model = load_checkpoint(checkpoint)
    tensor = process_image(image_path,device)
    probs, class_names = predict(image_path, model,top_k,device)
    console_display(image_path,probs,class_names,cat_to_name,device)
    
    
def console_display(image_path,probs,class_names ,cat_to_name,device):
    img = process_image(image_path,device)
    print(cat_to_name[image_path.split('/')[2]])
    print(probs)
    print(class_names)
        
    
def predict(image_path, model, top_k,device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    tensor = process_image(image_path,device)  
    with torch.set_grad_enabled(False):
        tensor = tensor.unsqueeze(0)
        output = model(tensor)
        probs, classes = F.softmax(output,dim = 1).topk(top_k)
        
    probs,classes = np.array(probs[0]).ravel(), np.array(classes[0].add(1)).ravel()   
    classes_names = np.array([cat_to_name[str(x)] for x in classes])
    
    return probs,classes_names

def set_device(enable_gpu = False) :
    if enable_gpu and torch.cuda.is_available() :
        device = torch.device("cuda:0")
    else :
        device = torch.device("cpu")
        
    return device    

    
    
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model = checkpoint['model']                                
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    

    
  

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", required=True,help="path to image")
    ap.add_argument("-c", "--checkpoint", required=True,help="saved model to use it in inference")
    ap.add_argument("-k", "--top_k", required=False,help="Choose architecture" ,type = int)  
    ap.add_argument("-cn", "--category_names", required=False,help="json file for category names ")
    ap.add_argument("-gp", "--gpu", required=False,help="Set device")
    args = vars(ap.parse_args())
    
    image_path = args['image_path']
    checkpoint = args['checkpoint']
   
        
    top_k = 5
    if args['top_k'] is not None:
        top_k = args['top_k']
    
    
    category_file = "cat_to_name.json"
    if args['category_names'] is not None:
        category_file = args['category_names']
        
    cat_to_name = {}    
    with open(category_file, 'r') as f:
        cat_to_name = json.load(f)    
            
        
    device = "cpu"
    if args['gpu'] is not None:
        device = set_device(enable_gpu = True)
        print("Setting device" , device)
        
        
    run(image_path,checkpoint,top_k,cat_to_name,device)
        
        
        
      