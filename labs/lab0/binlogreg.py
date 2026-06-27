from matplotlib import pyplot as plt
import numpy as np
import data

def binlogreg_train(X,Y_):
    '''
        Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

        Povratne vrijednosti
        w, b: parametri logističke regresije
    '''

    w = np.random.randn(X.shape[1], 1)
    b = 0

    param_niter = 1000
    param_delta = 0.01
    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b   # N x 1
        
        # vjerojatnosti razreda c_1
        probs = 1 / (1 + np.exp(-scores))     # N x 1

        # gubitak
        loss = -np.mean(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs))     # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_.reshape(-1, 1)  # N x 1
        
        # gradijenti parametara
        grad_w = np.dot(X.T, dL_dscores) / X.shape[0]     # D x 1
        grad_b = np.mean(dL_dscores)     # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    '''
        Argumenti
        X:  podatci, np.array NxD
        w, b: parametri logističke regresije

        Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''
    scores = np.dot(X, w) + b   # N x 1
    probs = 1 / (1 + np.exp(-scores))     # N x 1
    # u binlogreg_classify, na kraju:
    return probs.squeeze()

def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs > 0.5).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()

    # print(data.eval_AP([0,0,1,0,1,1]))
    # print(data.eval_AP([0,1,0,1,0,1]))
    # print(data.eval_AP([1,0,1,0,1,0]))
    # print(data.eval_AP([0,0,0,1,1,1]))

