## Steps

- Uninstall any prior `llava` installs in your chosen environment. Then run :

    `
        pip uninstall llava
        cd LLaVA-NeXT/
        pip install -e .
    `
- Uninstall `trl`, if present already. 
- Note that *DeepSpeed-ZeRO3 does not support quantization* (since it supports model parameter distribution over multiple GPUs, so that with quantization is complicated and not yet supported). So, if your model training doesn't fit with `training_args.bits=16 (default)`, you need to try out quantized fine-tuning. 

- For quantized (i.e., LoRA) fine-tuning with `training_args.bits=[4,8]`, use `zero2.json` instead.
- However, the previous setup passes a `device_map`, which requires `low_cpu_mem_usage=True`. So, set the parameter value accordingly.
