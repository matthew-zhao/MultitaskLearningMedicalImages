import torch
import torch.nn as nn
import json, os
from torch.nn.functional import softmax, relu, avg_pool2d, sigmoid

from sklearn.metrics import roc_auc_score, roc_curve, auc

from constants import CHEXPERT_LABEL_ORDERING
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

class MultiTaskSeparateAgent(BaseAgent):
    def __init__(self, num_classes, model, input_size, uncertainty_strategy='u_ones', task_prob=None):
        super(MultiTaskSeparateAgent, self).__init__()
        self.num_tasks = len(num_classes)
        self.task_prob = task_prob
        self.models = {
            study_type: model.to(self.device)
            for study_type, model in Model(num_tasks=num_classes, pretrained_model=model,
                input_size=input_size, uncertainty_strategy=uncertainty_strategy).items()
        }
        self.uncertainty_strategy = uncertainty_strategy

    def train_head(self, criterions, train_data, num_head_phases=5):
        for study_type, model in self.models.items():
            model.decoder.train()

        optimizers = {
            study_type: torch.optim.Adam(model.decoder.parameters(), lr=0.0001)
            for study_type, model in self.models.items()
        }

        for phase in range(num_head_phases):
            num_batches = 0
            for inputs, labels, level, study_type in train_data.get_loader(prob=self.task_prob if self.task_prob else 'uniform'):
                model = self.models[study_type]
                optimizer = optimizers[study_type]
                criterion = criterions[study_type]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches += 1

            print(num_batches)


    def train(self, criterions, train_data, test_data, num_phases=20, save_history=False, save_path='.', verbose=False):
        for study_type, model in self.models.items():
            model.train()

        optimizers = {
            study_type: torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
            for study_type, model in self.models.items()
        }
        # TODO: Eventually, we want to be able to customize whether to use LR scheduler and how
        schedulers = {
            study_type: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-3,
                threshold_mode='abs', patience=1, verbose=True)
            for study_type, optimizer in optimizers.items()
            if study_type != 'chexpert'
        }
        accuracy = []

        for phase in range(num_phases):
            num_batches = 0
            y_true_across_batches = []
            y_predict_across_batches = []

            y_true_per_task = {}
            y_predict_per_task = {}
            for inputs, labels, level, study_type in train_data.get_loader(prob=self.task_prob if self.task_prob else 'uniform'):
                model = self.models[study_type]
                optimizer = optimizers[study_type]
                criterion = criterions[study_type]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches += 1

                #_, predict_labels = torch.max(sigmoid(outputs).detach(), 1)
                if study_type != 'chexpert':
                    sigmoid_output = sigmoid(outputs).detach()
                    predict_labels = sigmoid_output[:,1]
                elif self.uncertainty_strategy == 'u_multiclass':
                    # for each observation:
                    # output probability of positive label (1.0) after applying softmax restricted to + and - classes
                    # how to know which class is which?
                    outputs_detached = outputs.detach()
                    softmax_output = softmax(torch.narrow(outputs_detached, 1, 0, 2), dim=1)
                    predict_labels = softmax_output[:,1,:]
                else:
                    sigmoid_output = sigmoid(outputs).detach()
                    predict_labels = sigmoid_output

                # UNCOMMENT IF WE NEED TO SOMEHOW CALCULATE AUC FOR JUST MURA OR JUST CHEXPERT
                # y_true_across_batches.append(labels)
                # y_predict_across_batches.append(predict_labels)

                if study_type in y_true_per_task and study_type in y_predict_per_task:
                    y_true_per_task[study_type].append(labels)
                    y_predict_per_task[study_type].append(predict_labels)
                else:
                    y_true_per_task[study_type] = [labels]
                    y_predict_per_task[study_type] = [predict_labels]

            if len(y_predict_across_batches) > 0:
                y_predicts = torch.cat(y_predict_across_batches)
                y_trues = torch.cat(y_true_across_batches)

                if study_type != 'chexpert':
                    area_under_curve = roc_auc_score(y_trues.cpu().numpy(), y_predicts.cpu().numpy())
                else:
                    # calculate roc_auc for each label
                    area_under_curve = [
                        roc_auc_score(y_trues[:,i].cpu().numpy(), y_predicts[:,i].cpu().numpy())
                        for i in range(y_predicts.size(1))
                    ]

            auc_per_task_training = {}
            if len(y_predict_per_task) > 0:
                for task in y_predict_per_task:
                    y_predicts = torch.cat(y_predict_per_task[task])
                    y_trues = torch.cat(y_true_per_task[task])

                    if task == 'chexpert':
                        auc_per_task_training[task] = {
                            CHEXPERT_LABEL_ORDERING[i]: roc_auc_score(y_trues[:,i].cpu().numpy(), y_predicts[:,i].cpu().numpy())
                            for i in range(y_predicts.size(1))
                        }
                        continue

                    auc_per_task_training[task] = {"abnormality": roc_auc_score(y_trues.cpu().numpy(), y_predicts.cpu().numpy())}


            last_phase = True if phase == (num_phases - 1) else False
            fpr, tpr, thresholds, auc, roc_curve_graphing_info_per_task, auc_per_task = self.eval(test_data, last_phase=last_phase)

            # only decay learning rate on plateau for MURA
            for study_type, scheduler in schedulers.items():
                scheduler.step(auc_per_task[study_type]['abnormality'])

            print(num_batches)

            if verbose:
                if area_under_curve and auc:
                    print('[Phase {}] Training AUC Overall: {}'.format(phase+1, area_under_curve))
                    print('[Phase {}] Validation AUC Overall: {}'.format(phase+1, auc))

                for task, label_type_to_auc in auc_per_task_training.items():
                    for label_type, auc in label_type_to_auc.items():
                        print('[Phase {}] [Task {}] [Label {}] Training AUC: {}'.format(phase+1, task, label_type, auc))

                for task, label_type_to_auc in auc_per_task.items():
                    for label_type, auc in label_type_to_auc.items():
                        print('[Phase {}] [Task {}] [Label {}] Validation AUC: {}'.format(phase+1, task, label_type, auc))

            if last_phase:
                if fpr and tpr:
                    print('False Positive Rates Overall: {}'.format(fpr))
                    print('True Positive Rates Overall: {}'.format(tpr))

                for task, label_type_to_roc_curve_graphing_info in roc_curve_graphing_info_per_task.items():
                    for label_type, (fpr, tpr, thresholds) in label_type_to_roc_curve_graphing_info:
                        print('[Task {}][Label {}] False Positive Rates: {}'.format(task, label_type, fpr))
                        print('[Task {}][Label {}] True Positive Rates: {}'.format(task, label_type, tpr))

        #if save_history:
        #    self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(history):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def eval(self, data, last_phase=False):
        correct = [0 for _ in range(self.num_tasks)]
        total = [0 for _ in range(self.num_tasks)]

        y_true_across_batches = []
        y_predict_across_batches = []

        y_true_per_task = {}
        y_predict_per_task = {}

        y_levels = []

        prev_task = 0

        with torch.no_grad():
            for study_type, model in self.models.items():
                model.eval()

            valid_loss = 0.
            for inputs, labels, level, study_type in data.get_loader():
                model = self.models[study_type]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                #loss = self.criterion(outputs, labels)
                #valid_loss += loss.item()
                # average across views
                # _, per_view_predict_labels = torch.max(outputs.detach(), 1)

                if study_type != 'chexpert':
                    output = torch.mean(sigmoid(outputs).detach(), 0, keepdim=True)
                    predict_labels = output[:,1]
                elif self.uncertainty_strategy == 'u_multiclass':
                    # for each observation:
                    # output probability of positive label (1.0) after applying softmax restricted to + and - classes
                    # how to know which class is which?
                    outputs_detached = outputs.detach()
                    softmax_output = softmax(torch.narrow(outputs_detached, 1, 0, 2), dim=1)
                    predict_labels = softmax_output[:,1,:]
                else:
                    output = sigmoid(outputs).detach()
                    predict_labels = output
                # _, predict_labels = torch.max(output_avg_views, 1)

                # UNCOMMENT IF WE NEED TO SOMEHOW CALCULATE AUC FOR JUST MURA VS CHEXPERT
                # y_true_across_batches.append(labels)
                # y_predict_across_batches.append(predict_labels)

                # scalable to ensure that this only applies to MURA (?)
                if not study_type:
                    raise ValueError("Don't know what study type to evaluate model on")

                if study_type in y_true_per_task and study_type in y_predict_per_task:
                    y_true_per_task[study_type].append(labels)
                    y_predict_per_task[study_type].append(predict_labels)
                else:
                    y_true_per_task[study_type] = [labels]
                    y_predict_per_task[study_type] = [predict_labels]

                #total[task] += labels.size(0)
                #correct[task] += (per_view_predict_labels == labels).sum().item()

            fpr, tpr, thresholds = None, None, None
            area_under_curve = None
            if len(y_predict_across_batches) > 0:
                y_predicts = torch.cat(y_predict_across_batches)
                y_trues = torch.cat(y_true_across_batches)

                area_under_curve = roc_auc_score(y_trues.cpu().numpy(), y_predicts.cpu().numpy())
                if last_phase:
                    fpr, tpr, thresholds = roc_curve(y_trues.cpu().numpy(), y_predicts.cpu().numpy())

            roc_curve_graphing_info_per_task = {}
            auc_per_task = {}
            if len(y_predict_per_task) > 0:
                for task in y_predict_per_task:
                    y_predicts = torch.cat(y_predict_per_task[task])
                    y_trues = torch.cat(y_true_per_task[task])

                    if task == 'chexpert':
                        if last_phase:
                            roc_curve_graphing_info_per_task[task] = {
                                CHEXPERT_LABEL_ORDERING[i]: roc_curve(y_trues[:,i].cpu().numpy(), y_predicts[:,i].cpu().numpy())
                                for i in range(y_predicts.size(1))
                            }
                        auc_per_task[task] = {
                            CHEXPERT_LABEL_ORDERING[i]: roc_auc_score(y_trues[:,i].cpu().numpy(), y_predicts[:,i].cpu().numpy())
                            for i in range(y_predicts.size(1))
                        }
                        continue

                    if last_phase:
                        roc_curve_graphing_info_per_task[task] = {"abnormality": roc_curve(y_trues.cpu().numpy(), y_predicts.cpu().numpy())}
                    auc_per_task[task] = {"abnormality": roc_auc_score(y_trues.cpu().numpy(), y_predicts.cpu().numpy())}

            # letting model know it's back to training time
            for study_type, model in self.models.items():
                model.train()

            #return [c / t for c, t in zip(correct, total)]
            return fpr, tpr, thresholds, area_under_curve, roc_curve_graphing_info_per_task, auc_per_task


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for study_type, model in self.models.items():
            filename = os.path.join(save_path, 'model_{}'.format(study_type))
            torch.save(model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            for study_type, model in self.models.items():
                filename = os.path.join(save_path, 'model_{}'.format(study_type))
                model.load_state_dict(torch.load(filename))