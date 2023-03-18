import wandb

    
class WanDB():
    def __init__(self, project_name, entity='teddynote'):
        wandb.init(project=project_name, entity=entity)
        wandb.define_metric('loss', summary='min')
        wandb.define_metric('acc', summary='max')
        wandb.define_metric('f1', summary='max')
        
    def set_config(self, configs):
        wandb.config = configs
        
    def log(self, items, step=None):
        wandb.log(items, step=step)
        text_builder = []
        if step is not None:
            text_builder.append(f'epoch {step+1:02d}')
            
        for k, v in items.items():
            text_builder.append(f'{k}: {v:.5f}')
        
        print(', '.join(text_builder))
        
        
    def alert(self, title, text):
        wandb.alert(title=title, text=text)
        

class Logger():
    def __init__(self, project_name):
        self.project_name = project_name
        self.configs = None
        
    def set_config(self, configs):
        self.configs = configs
        
    def log(self, step=None, **kwargs):
        text_builder = []
        if step is not None:
            text_builder.append(f'epoch {step+1:02d}')
            
        for k, v in kwargs.items():
            text_builder.append(f'{k}: {v:.5f}')
        
        print(', '.join(text_builder))
        
        
        