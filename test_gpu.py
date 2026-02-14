import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2,2, device="cuda")
    print("ok:", x)
