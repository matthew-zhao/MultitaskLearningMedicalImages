import torch
import torch.nn as nn
import json, os

from sklearn.metrics import roc_auc_score

from model import Model

class BaseAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data, test_data, num_epochs, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save_model(self, save_path):
        pass

    def load_model(self, save_path):
        pass

class SingleTaskAgent(BaseAgent):
    def __init__(self, num_classes, model):
        super(SingleTaskAgent, self).__init__()
        self.model = Model(num_tasks=num_classes, pretrained_model=model).to(self.device)


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(history, f)


    def eval(self, data):
        correct = 0
        total = 0

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)
                correct += (predict_labels == labels).sum().item()

            self.model.train()

            return correct / total


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'model')

        torch.save(self.model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            filename = os.path.join(save_path, 'model')
            self.model.load_state_dict(torch.load(filename))


class StandardAgent(SingleTaskAgent):
    def __init__(self, num_classes_single, num_classes_multi, multi_task_type, model):
        if multi_task_type == 'binary':
            super(StandardAgent, self).__init__(num_classes=num_classes_single, model=model)
            self.eval = self._eval_binary
            self.num_classes = num_classes_single
        elif multi_task_type == 'multiclass':
            super(StandardAgent, self).__init__(num_classes=num_classes_single, model=model)
            self.eval = self._eval_multiclass
            self.num_classes = num_classes_multi
        else:
            raise ValueError('Unknown multi-task type: {}'.format(multi_task_type))


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def _eval_binary(self, data):
        correct = [0 for _ in range(self.num_classes)]
        total = 0

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in data.get_loader():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)

                for c in range(self.num_classes):
                    correct[c] += ((predict_labels == c) == (labels == c)).sum().item()

            self.model.train()

            return [c / total for c in correct]


    def _eval_multiclass(self, data):
        num_tasks = len(self.num_classes)
        correct = [0 for _ in range(num_tasks)]
        total = [0 for _ in range(num_tasks)]

        with torch.no_grad():
            self.model.eval()

            for t in range(num_tasks):
                task_labels = data.get_labels(t)
                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predict_labels = torch.max(outputs[:, task_labels].detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

            self.model.train()

            return [c / t for c, t in zip(correct, total)]

class MultiTaskSeparateAgent(BaseAgent):
    def __init__(self, num_classes, model, input_size, task_prob=None):
        super(MultiTaskSeparateAgent, self).__init__()
        self.num_tasks = len(num_classes)
        self.task_prob = task_prob
        self.models = [model.to(self.device) for model in Model(num_tasks=num_classes, pretrained_model=model, input_size=input_size)]


    def train(self, criterions, train_data, test_data, num_phases=50, save_history=False, save_path='.', verbose=False):
        for model in self.models:
            model.train()

        #optimizers = [optim.SGD(model.parameters(), lr=0.001) for model in self.models]
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.0001) for model in self.models]
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, verbose=True) for optimizer in optimizers]
        accuracy = []

        for phase in range(num_phases):
            num_batches = 0
            #prev_task = 0
            for inputs, labels, task in train_data.get_loader(prob=self.task_prob if self.task_prob else 'uniform'):
                # if task != prev_task:
                #     prev_task = task
                model = self.models[task]
                optimizer = optimizers[task]
                criterion = criterions[task]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches += 1

            val_accuracy = self.eval(test_data)
            for scheduler in schedulers:
                scheduler.step(val_accuracy[0])
            accuracy.append(val_accuracy)

            print(num_batches)

            if verbose:
                print('[Phase {}] Accuracy: {}'.format(phase+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def eval(self, data):
        correct = [0 for _ in range(self.num_tasks)]
        total = [0 for _ in range(self.num_tasks)]

        y_true_across_batches = []
        y_predict_across_batches = []

        prev_task = 0

        with torch.no_grad():
            #for t, model in enumerate(self.models):
            # letting model know it's eval time
            for model in self.models:
                model.eval()

            for inputs, labels, task in data.get_loader():
                model = self.models[task]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                # average across views
                # _, per_view_predict_labels = torch.max(outputs.detach(), 1)

                output_avg_views = torch.mean(outputs.detach(), 0, keepdim=True)
                _, predict_labels = torch.max(output_avg_views, 1)
                #print(predict_labels, type(predict_labels))
                #print(labels, type(labels))
                y_true_across_batches.append(labels)
                y_predict_across_batches.append(predict_labels)

                #total[task] += labels.size(0)
                #correct[task] += (per_view_predict_labels == labels).sum().item()

            if len(y_predict_across_batches) > 0:
                y_predicts = torch.cat(y_predict_across_batches)
                y_trues = torch.cat(y_true_across_batches)

                area_under_curve = roc_auc_score(y_trues.cpu().numpy(), y_predicts.cpu().numpy())

            # letting model know it's back to training time
            for model in self.models:
                model.train()

            #return [c / t for c, t in zip(correct, total)]
            return [area_under_curve]


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for t, model in enumerate(self.models):
            filename = os.path.join(save_path, 'model{}'.format(t))
            torch.save(model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            for t, model in enumerate(self.models):
                filename = os.path.join(save_path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))