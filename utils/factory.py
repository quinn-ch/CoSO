def get_model(model_name, args):
    name = model_name.lower()
    if name == "coso":
        from models.coso import Learner
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. This repository only keeps the CoSO implementation.")
    return Learner(args)
