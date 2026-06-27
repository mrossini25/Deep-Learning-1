import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

class Random2DGaussian():
    def __init__(self):
        self.minx = 0
        self.maxx = 10
        self.miny = 0
        self.maxy = 10
        self.mean = np.random.random_sample(2) * (self.maxx - self.minx) + self.minx
        eigvalx = (np.random.random_sample() * (self.maxx - self.minx) / 5)**2
        eigvaly = (np.random.random_sample() * (self.maxy - self.miny) / 5)**2
        self.D = np.array([[eigvalx, 0], [0, eigvaly]])
        phi = np.random.random_sample() * 2 * np.pi
        self.R = np.array([[np.cos(phi), -np.sin(phi)],
                           [np.sin(phi), np.cos(phi)]])
        self.cov = self.R.T @ self.D @ self.R

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, n)
    

def sample_gauss_2d(C, N):
    '''
        Argumenti
        C:  broj slučajnih bivarijatnih Gaussovih razdioba, int
        N:  broj uzoraka

        Povratne vrijednosti
        X:  uzorci, np.array Nx2
    '''
    Gs=[]
    Ys=[]
    for i in range(C):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_= np.hstack([[Y]*N for Y in Ys])
  
    return X,Y_

def eval_perf_binary(Y,Y_):
    '''
        Argumenti
        Y:  predviđeni indeksi razreda, np.array Nx1
        Y_: stvarni indeksi razreda, np.array Nx1

        Povratne vrijednosti
        acc: točnost klasifikacije
        recall: odziv klasifikacije
        prec: preciznost klasifikacije
    '''
    acc = np.mean(Y == Y_)
    recall = np.mean((Y == 1) & (Y_ == 1)) / np.mean(Y_ == 1)
    prec = np.mean((Y == 1) & (Y_ == 1)) / np.mean(Y == 1)
    return acc, recall, prec

def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_)+1
    M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
    for i in range(n):
        tp_i = M[i,i]
        fn_i = np.sum(M[i,:]) - tp_i
        fp_i = np.sum(M[:,i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append( (recall_i, precision_i) )
    
    accuracy = np.trace(M)/np.sum(M)
    
    return accuracy, pr, M

def eval_AP(Yr):
    '''
        Argumenti
        Yr: rangiranu listu točnih razreda, np.array Nx1

        Povratne vrijednosti
        AP: prosječna preciznost klasifikacije
    '''
    n = len(Yr)
    pos = sum(Yr)
    neg = n - pos
    
    tp = pos
    tn = 0
    fn = 0
    fp = neg
    
    sumprec = 0
    for x in Yr:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)    

        if x:
            sumprec += precision

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec/pos

def graph_data(X, Y_, Y, special=[]):
    '''
    X  ... podatci (np.array dimenzija Nx2)
    Y_ ... točni indeksi razreda podataka (Nx1)
    Y  ... predviđeni indeksi razreda podataka (Nx1)
    special ... lista indeksa specijalnih točaka
    '''
    palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    for i in range(len(palette)):
        colors[Y_==i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40
    
    # draw the correctly classified datapoints
    good = (Y_==Y)
    plt.scatter(X[good,0],X[good,1], c=colors[good], 
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_!=Y)
    plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
                s=sizes[bad], marker='s', edgecolors='black')


def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

def graph_surface(function, rect, offset=0.5, width=256, height=256):
    '''
  function... decizijska funkcija (Nx2)->(Nx1)
  rect   ... željena domena prikaza zadana kao:
             ([x_min,y_min], [x_max,y_max])
  offset ... "nulta" vrijednost decizijske funkcije na koju 
             je potrebno poravnati središte palete boja;
             tipično imamo:
             offset = 0.5 za probabilističke modele 
                (npr. logistička regresija)
             offset = 0 za modele koji ne spljošćuju
                klasifikacijske mjere (npr. SVM)
  width,height ... rezolucija koordinatne mreže
    '''
    lsw = np.linspace(rect[0][1], rect[1][1], width) 
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0,xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

    #get the values and reshape them
    values=function(grid).reshape((width,height))
    
    # fix the range and offset
    delta = offset if offset else 0
    maxval=max(np.max(values)-delta, - (np.min(values)-delta))
    
    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, 
        vmin=delta-maxval, vmax=delta+maxval)
        
    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])

if __name__ == "__main__":
    np.random.seed(100)
    # G=Random2DGaussian()
    # print("Mean:", G.mean)
    # print("Covariance:", G.cov)
    # X=G.get_sample(100)
    #plt.scatter(X[:,0], X[:,1])
    #plt.show()

    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)
  
    # get the class predictions
    Y = myDummyDecision(X)>0.5
    
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0)
    
    # graph the data points
    graph_data(X, Y_, Y)
  
    # show the results
    plt.show()