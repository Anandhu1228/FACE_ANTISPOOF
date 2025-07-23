import torch
import torch.onnx
import timm
import os

def convert_efficientnet_to_onnx():
    model_path = "models/best_class_3.pth"
    output_path = "models/best_class_3.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return False
    
    print(f"Loading model from: {model_path}")
    
    model = timm.create_model('efficientnet_b0', num_classes=2, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,                         
        dummy_input,
        output_path, 
        export_params=True,  
        opset_version=11,    
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'], 
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model successfully converted to ONNX!")
    print(f"ONNX model saved at: {output_path}")
    
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed!")
    
    return True

if __name__ == "__main__":
    try:
        convert_efficientnet_to_onnx()
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")


