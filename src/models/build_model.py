from config.configurator import configs
import importlib

def build_model(data_handler):
    
    model_name = configs['model']['name']

    # search for models in sub-directories
    module_path = ""
    candidates = [".".join(['models', model_name])]
    # for sub_dir in ['code', 'chat', 'nl']:
    #     candidates.append(".".join(['models', sub_dir, model_name]))
    for candidate in candidates:
        if importlib.util.find_spec(candidate) is not None:
            module_path = candidate
            break
    if module_path == "":
        raise NotImplementedError(f'Model {model_name} is not implemented. Search path: {candidates}')
    
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == model_name.lower():
            return getattr(module, attr)(data_handler)
    else:
        raise NotImplementedError('Model Class {} is not defined in {}'.format(model_name, module_path))
