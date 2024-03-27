import sys
from util.conf import OptionConf
from data.loader import FileIO
from time import strftime, localtime, time


class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for compatibility with Python 3.
        self.terminal.flush()
        self.log.flush()

class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        log_filepath = OptionConf(self.config['output.setup'])['-dir'] + self.config['model.name'] + '@' + current_time + '.log'
        sys.stdout = DualOutput(log_filepath)
        # sys.stdout = open(log_filepath, 'w')
        # sys.stdout = sys.__stdout__
        print('SEED: ', self.seed)
        eval(recommender).execute()
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
