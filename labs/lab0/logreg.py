from matplotlib import pyplot as plt
import numpy as np
import data

def logreg_train(X,Y_):
    # pri računanju softmaksa obratite pažnju
    # na odjeljak 4.1 udžbenika
    # (Deep Learning, Goodfellow et al)!

    W = np.random.randn(X.shape[1], Y_.shape[1])
    b = np.zeros((1, Y_.shape[1]))

    param_niter = 10000
    param_delta = 0.1

    for i in range(param_niter):
        scores = np.dot(X, W) + b   # N x C
        expscores = np.exp(scores)  # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss  = -np.mean(np.sum(Y_ * logprobs, axis=1))      # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_  # N x C

        # gradijenti parametara
        grad_W = np.dot(X.T, dL_ds) / X.shape[0]    # D x C
        grad_b = np.mean(dL_ds, axis=0, keepdims=True)    # 1 x C

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b
    
    return W, b


def logreg_classify(X, W, b):
    scores = np.dot(X, W) + b   # N x C
    expscores = np.exp(scores)  # N x C

    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1

    # vjerojatnosti razreda 
    probs = expscores / sumexp     # N x C
    return probs

def logreg_decfun(W, b):
    def classify(X):
        return logreg_classify(X, W, b)
    return classify

if __name__=="__main__":
    np.random.seed(100)

    C = 3
    X, Y_ = data.sample_gauss_2d(C, 100)

    # konverzija u one-hot za logreg_train
    Y_oh = np.zeros((len(Y_), C))
    Y_oh[np.arange(len(Y_)), Y_] = 1

    W, b = logreg_train(X, Y_oh)
    probs = logreg_classify(X, W, b)   # N x C
    Y = np.argmax(probs, axis=1)       # predviđeni razred

    accuracy, precision, recall = data.eval_perf_multi(Y, Y_)
    print(f"Višeklasni: accuracy={accuracy:.3f}")

    decfun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda X: np.argmax(decfun(X), axis=1), bbox, offset=None)
    data.graph_data(X, Y_, Y)
    plt.show()