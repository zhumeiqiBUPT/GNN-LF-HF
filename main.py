import os
import pandas as pd
import argparse
from utils import *
from model import GNNs
from training import train_model
from earlystopping import stopping_args
from propagation import *
from load_data import *



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=20)
    parse.add_argument("-t", "--type", help="model for training, (PPNP=0, GNN-LF=1, GNN-HF=2)", type=int, required=True)
    parse.add_argument("-f", "--form", help="closed/iter form models (closed=0, iterative=1)", type=int, required=True)
    parse.add_argument("--device", help="GPU device", type=str, default="0")
    parse.add_argument("--niter", help="times for iteration", type=int, default=10)
    parse.add_argument("--reg_lambda", help="regularization", type=float, default=5e-3)
    parse.add_argument("--lr", help="learning rate", type=float, default=0.01)
    args = parse.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.dataset == 'acm':
        graph, idx_np = load_new_data_acm(args.labelrate)
    elif args.dataset == 'wiki':
        graph, idx_np = load_new_data_wiki(args.labelrate)
        args.lr = 0.03
        args.reg_lambda = 5e-4
    elif args.dataset == 'ms':
        graph, idx_np = load_new_data_ms(args.labelrate)
    else:
        if args.dataset == 'cora':
            feature_dim = 1433
        elif args.dataset == 'citeseer':
            feature_dim = 3703
        elif args.dataset == 'pubmed':
            feature_dim = 500
        graph, idx_np = load_new_data_tkipf(args.dataset, feature_dim, args.labelrate)


    print_interval = 100
    device = 'cuda'
    test = True

    propagation = []
    
    para_list = [0.9]

    for para1 in [0.1]:
        for para2 in para_list:
            if args.type == 0:
                model_type="PPNP"
                if args.form == 0:
                    model_form = "closed"
                    propagation = PPRExact(graph.adj_matrix, alpha=para1)
                else:
                    model_form = "itera"
                    propagation = PPRPowerIteration(graph.adj_matrix, alpha=para1, niter=args.niter)
            elif args.type == 1:
                model_type = "GNN-LF"
                if args.form == 0:
                    model_form = "closed"
                    propagation = LFExact(graph.adj_matrix, alpha=para1, mu=para2)
                else:
                    model_form = "itera"
                    propagation = LFPowerIteration(graph.adj_matrix, alpha=para1, mu=para2, niter=args.niter)
            elif args.type == 2:
                model_type = "GNN-HF"
                if args.form == 0:
                    model_form = "closed"
                    propagation = HFExact(graph.adj_matrix, alpha=para1, beta=para2)
                else:
                    model_form = "itera"
                    propagation = HFPowerIteration(graph.adj_matrix, alpha=para1, beta=para2, niter=args.niter)

            model_args = {
                'hiddenunits': [64],
                'drop_prob': 0.5,
                'propagation': propagation}

            results = []

            i_tot = 0
            average_time = 10
            for _ in range(average_time):
                i_tot += 1
                logging_string = f"Iteration {i_tot} of {average_time}"
                print(logging_string)
                _, result = train_model(idx_np,  args.dataset, GNNs, graph, model_args, args.lr, args.reg_lambda,
                        stopping_args, test, device, None, print_interval)
                results.append({})
                results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
                results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
                results[-1]['runtime'] = result['runtime']
                results[-1]['runtime_perepoch'] = result['runtime_perepoch']

            result_df = pd.DataFrame(results)
            result_df.head()

            stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
            valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
            runtime = calc_uncertainty(result_df['runtime'])
            runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])

            f = open(str(args.dataset) + '_labelrate_' + str(args.labelrate) + '_model_' + str(model_type) + '_form_' + str(model_form) + '.txt','a+')

            print("model_" + str(model_type) + "_form_" + str(model_form)  + "\n" 
                  "Early stopping: Accuracy: {:.2f} ± {:.2f}%\n" 
                  "{}: ACC: {:.2f} ± {:.2f}%\n"
                  "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms\n"
                  .format(
                      stopping_acc['mean'] * 100,
                      stopping_acc['uncertainty'] * 100,
                      'Test' if test else 'Validation',
                      valtest_acc['mean'] * 100,
                      valtest_acc['uncertainty'] * 100,
                      runtime['mean'],
                      runtime['uncertainty'],
                      runtime_perepoch['mean'] * 1e3,
                      runtime_perepoch['uncertainty'] * 1e3,
                  ))


            f.write("\nmodel_" + str(model_type) + "_form_" + str(model_form)  + "\n" 
                  "Early stopping: Accuracy: {:.2f} ± {:.2f}%\n"
                  "{}: ACC: {:.2f} ± {:.2f}%\n"
                  "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms\n\n"
                  .format(
                      stopping_acc['mean'] * 100,
                      stopping_acc['uncertainty'] * 100,
                      'Test' if test else 'Validation',
                      valtest_acc['mean'] * 100,
                      valtest_acc['uncertainty'] * 100,
                      runtime['mean'],
                      runtime['uncertainty'],
                      runtime_perepoch['mean'] * 1e3,
                      runtime_perepoch['uncertainty'] * 1e3,
                  ))

