import pandas as pd
import numpy as np

class LRWGD:
    def __init__(self):
        self.W = np.random.normal(size=5)
        self.b = 0
    def __linear_regression(self,X):
        return self.W[0]*X[:,0]+self.W[1]*X[:,1]+self.W[2]*X[:,2]+self.W[3]*X[:,3]+self.W[4]*X[:,4]+self.b
    def __loss_function(self,yTrue,yPred):
        return np.mean((yTrue-yPred)**2)
    def __gradient_descent(self,X,y,learning_rate):
        dldw = np.zeros(5)
        dldb = 0
        for i in range(X.shape[0]):
            dldw[0] += 2*(y[i]-(self.W[0]*X[i][0]+self.W[1]*X[i][1]+self.W[2]*X[i][2]+self.W[3]*X[i][3]+self.W[4]*X[i][4]+self.b))*(-X[i][0])
            dldw[1] += 2*(y[i]-(self.W[0]*X[i][0]+self.W[1]*X[i][1]+self.W[2]*X[i][2]+self.W[3]*X[i][3]+self.W[4]*X[i][4]+self.b))*(-X[i][1])
            dldw[2] += 2*(y[i]-(self.W[0]*X[i][0]+self.W[1]*X[i][1]+self.W[2]*X[i][2]+self.W[3]*X[i][3]+self.W[4]*X[i][4]+self.b))*(-X[i][2])
            dldw[3] += 2*(y[i]-(self.W[0]*X[i][0]+self.W[1]*X[i][1]+self.W[2]*X[i][2]+self.W[3]*X[i][3]+self.W[4]*X[i][4]+self.b))*(-X[i][3])
            dldw[4] += 2*(y[i]-(self.W[0]*X[i][0]+self.W[1]*X[i][1]+self.W[2]*X[i][2]+self.W[3]*X[i][3]+self.W[4]*X[i][4]+self.b))*(-X[i][4])
            dldb += 2*(y[i]-(self.W[0]*X[i][0]+self.W[1]*X[i][1]+self.W[2]*X[i][2]+self.W[3]*X[i][3]+self.W[4]*X[i][4]+self.b))*(-1)
        self.W[0] = self.W[0] - learning_rate*(dldw[0]/self.N)
        self.W[1] = self.W[1] - learning_rate*(dldw[1]/self.N)
        self.W[2] = self.W[2] - learning_rate*(dldw[2]/self.N)
        self.W[3] = self.W[3] - learning_rate*(dldw[3]/self.N)
        self.W[4] = self.W[4] - learning_rate*(dldw[4]/self.N)
        self.b = self.b - learning_rate*(dldb/self.N)
    def cocokkan(self,X,y,learning_rate=0.01):
        n_epoch=1000
        self.N = X.shape[0]
        losses = np.zeros(n_epoch)
        for epoch in range(n_epoch):
            self.__gradient_descent(X,y,learning_rate)
            yHat = self.__linear_regression(X)
            losses[epoch] = self.__loss_function(y,yHat)
        return losses
    def predict(self,X):
        return self.__linear_regression(X)
    
class LRWRGD:
    def __init__(self):
        self.W = np.random.normal(size=5)  # Weight (5 fitur)
        self.b = 0  # Bias

    def __linear_regression(self, X):
        return np.dot(X, self.W) + self.b

    def __loss_function(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def __find_min(self, i, X, y):
        if i == 0:
            error = y[i] - (self.W[0] * X[i][0] + self.W[1] * X[i][1] +
                            self.W[2] * X[i][2] + self.W[3] * X[i][3] +
                            self.W[4] * X[i][4] + self.b)
            dldw0 = 2 * error * (-X[i][0])
            dldw1 = 2 * error * (-X[i][1])
            dldw2 = 2 * error * (-X[i][2])
            dldw3 = 2 * error * (-X[i][3])
            dldw4 = 2 * error * (-X[i][4])
            dldb = 2 * error * (-1)
            return dldw0, dldw1, dldw2, dldw3, dldw4, dldb
        else:
            dldw0, dldw1, dldw2, dldw3, dldw4, dldb = self.__find_min(i - 1, X, y)
            error = y[i] - (self.W[0] * X[i][0] + self.W[1] * X[i][1] +
                            self.W[2] * X[i][2] + self.W[3] * X[i][3] +
                            self.W[4] * X[i][4] + self.b)

            dldw0 += 2 * error * (-X[i][0])
            dldw1 += 2 * error * (-X[i][1])
            dldw2 += 2 * error * (-X[i][2])
            dldw3 += 2 * error * (-X[i][3])
            dldw4 += 2 * error * (-X[i][4])
            dldb += 2 * error * (-1)

            return dldw0, dldw1, dldw2, dldw3, dldw4, dldb

    def __gradient_descent_recursive(self, X, y, learning_rate):
        dldw = np.zeros(5)
        dldb = 0
        dldw[0], dldw[1], dldw[2], dldw[3], dldw[4], dldb = self.__find_min(X.shape[0] - 1, X, y)

        self.W -= learning_rate * (dldw / self.N)
        self.b -= learning_rate * (dldb / self.N)

    def cocokkan(self, X, y, learning_rate=0.01):
        n_epochs=1000
        self.N = X.shape[0]
        losses = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            self.__gradient_descent_recursive(X, y, learning_rate)
            y_pred = self.__linear_regression(X)
            losses[epoch] = self.__loss_function(y, y_pred)

        return losses

    def prediksi(self, X):
        return self.__linear_regression(X)
    
def load_model_wb(path):
    wb_df = pd.read_csv(path)
    return wb_df.values[0,:5],wb_df.values[0,5]


def main():
    model_r = LRWRGD()
    model_i = LRWGD()

    Wi,bi = load_model_wb('wb_iteratif.csv')
    Wr,br = load_model_wb('wb_rekursif.csv')
    model_i.W, model_i.b = Wi,bi
    model_r.W,model_r.b = Wr,br

    print("Prediksi Frekuensi Meminum Minuman Beralkohol")
    print("="*50)
    mcv = float(input('Masukan hasil mean corpusular volume: '))
    alkphos = float(input('Masukan hasil tes alkaline phosphotase: '))
    sgpt = float(input('Masukan hasil tes alanine aminotransferase: '))
    sgot = float(input('Masukan hasil tes aspartate aminotransferase: '))
    gammagt = float(input('Masukan hasil tes gamma-glutamyl transpeptidase: '))

    X = np.array([[mcv,alkphos,sgpt,sgot,gammagt]])
    Max = np.array([103,138,155,82,297])
    Min = np.array([65,23,4,5,5])
    X_norm = (X-Min)/(Max-Min)

    pred_i = model_i.predict(X_norm)
    pred_r = model_r.prediksi(X_norm)

    print("="*50)
    print(f'Hasil prediksi menggunakan model dengan skema iteratif: [{int(pred_i[0])}]')
    print("="*50)
    print(f'Hasil prediksi menggunakan model dengan skema rekursif: [{int(pred_r[0])}]')

main()
