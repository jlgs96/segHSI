
    python -u get_params.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.5   --idtest 0 
    python -u get_params.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 0.5 --use_boxconv   --idtest 0

    python -u get_params.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 1   --idtest 0 
    python -u get_params.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 1 --use_boxconv   --idtest 0 

    python -u get_params.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 2   --idtest 0 
    python -u get_params.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 2 --use_boxconv   --idtest 0 

