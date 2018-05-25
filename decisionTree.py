import sys
from math import log
import numpy as np

csv_f_1 = sys.argv[1]
csv_f_2 = sys.argv[2]
dp = sys.argv[3]
label_1 = sys.argv[4]
label_2 = sys.argv[5]
metrics = sys.argv[6]

db = np.genfromtxt(csv_f_1, dtype=None, delimiter=',')
db_t = np.genfromtxt(csv_f_2, dtype=None, delimiter=',')
tree = []

dp = int(dp)
if dp > len(db[0]) - 1:
    dp = len(db[0]) - 1


class DecisionTree:
    def __init__(self):
        self.depth = dp

        self.left = None
        self.right = None

        self.root_node = ""
        self.attr_val = ""
        self.label_y = ""
        self.label_y_nr = 1
        self.label_n = ""
        self.label_n_nr = 1
        self.line_number = 0

        self.train_data = np.delete(db, 0, axis=0)

        self.attr_name = db[0, :]
        self.attr_nr = 0
        self.attr_val_0 = []
        self.attr_val_1 = []

        self.general_label_0_nr = 0
        self.general_label_1_nr = 0

        self.node_nr = 0
        self.label = ""
        self.learned_tree = ""
        self.branch = ""

    def entropy(self, x, y):  # Calculate Entropy
        if x == 0 or y == 0:
            return 0
        else:
            s = x+y
            div_1 = float(x)/s
            div_2 = float(y)/s
            return - div_1*log(div_1, 2) - div_2*log(div_2, 2)

    def info_gain(self, a, b, c, d):  # Calculate Information Gain
        s1 = a + c
        s2 = b + d
        s3 = a + b
        s4 = c + d
        s5 = s3 + s4
        return self.entropy(s1, s2) - float(s3) / s5 * self.entropy(a, b) - float(s4) / s5 * self.entropy(c, d)

    def generate_attr_val(self):  # Get the Value of Different Attributes (including the Label Contents)
        for i in range(0, len(db[1])):
            self.attr_val_0.append(db[1, i])
            j = 2
            k = 0
            while k == 0:
                if db[j][i] != db[1][i]:
                    self.attr_val_1.append(db[j, i])
                    k += 1
                j += 1

    def generate_stump(self):  # Generate the Stump of the Tree
        for i in range(1, len(db)):
            if db[i, -1] == self.attr_val_0[-1]:
                self.general_label_0_nr += 1
            else:
                self.general_label_1_nr += 1

        self.learned_tree = '[' + str(self.general_label_0_nr) + ' ' + self.attr_val_0[-1] + '/'\
                            + str(self.general_label_1_nr) + ' ' + self.attr_val_1[-1] + ']' + '\n'

    def generate_tree(self):  # Generate the Whole Tree
        if self.depth == 0:  # Solve the Problem if the depth is equal to zero.
            if self.general_label_0_nr > self.general_label_1_nr:
                self.label = self.attr_val_0[-1]
            else:
                self.label = self.attr_val_1[-1]
        else:
            if len(self.train_data) != len(db)-1:
                self.branch = self.line_number*'| ' + self.root_node + ' = '\
                                    + self.attr_val + ': [' + str(self.label_y_nr) + ' ' + self.label_y + '/'\
                                    + str(self.label_n_nr) + ' ' + self.label_n + ']' + '\n'
                tree.append(self.branch)

            if self.line_number != self.depth and self.label_y_nr != 0 and self.label_n_nr != 0:
                info = []
                A = []
                B = []
                C = []
                D = []
                for i in range(0, len(self.attr_name)-1):
                    if i == self.attr_nr:
                        info.append(0)
                        A.append(0)
                        B.append(0)
                        C.append(0)
                        D.append(0)
                    else:
                        a = 0
                        b = 0
                        c = 0
                        d = 0
                        for j in range(0, len(self.train_data)):
                            if self.train_data[j][i] == self.attr_val_0[i]:

                                if self.train_data[j][-1] == self.attr_val_0[-1]:
                                    a += 1
                                else:
                                    b += 1
                            else:
                                if self.train_data[j][-1] == self.attr_val_0[-1]:
                                    c += 1
                                else:
                                    d += 1
                        ig = self.info_gain(a, b, c, d)
                        info.append(ig)
                        A.append(a)
                        B.append(b)
                        C.append(c)
                        D.append(d)

                for i in range(0, len(info)):
                    if info[i] == max(info):
                        self.node_nr = i

                        self.left = DecisionTree()
                        self.right = DecisionTree()

                        self.left.train_data = self.train_data[self.train_data[:, i] == self.attr_val_0[i]]
                        self.left.root_node = self.attr_name[i]
                        self.left.attr_nr = i
                        self.left.label_y = self.attr_val_0[-1]
                        self.left.label_y_nr = A[i]
                        self.left.label_n = self.attr_val_1[-1]
                        self.left.label_n_nr = B[i]
                        self.left.line_number = self.line_number + 1
                        self.left.attr_val = self.attr_val_0[i]
                        self.left.attr_val_0 = self.attr_val_0
                        self.left.attr_val_1 = self.attr_val_1

                        self.left.generate_tree()

                        self.right.train_data = self.train_data[self.train_data[:, i] == self.attr_val_1[i]]
                        self.right.root_node = self.left.root_node
                        self.right.attr_nr = i
                        self.right.label_y = self.attr_val_0[-1]
                        self.right.label_y_nr = C[i]
                        self.right.label_n = self.attr_val_1[-1]
                        self.right.label_n_nr = D[i]
                        self.right.line_number = self.line_number + 1
                        self.right.attr_val = self.attr_val_1[i]
                        self.right.attr_val_0 = self.attr_val_0
                        self.right.attr_val_1 = self.attr_val_1

                        self.right.generate_tree()
            else:
                if self.label_y_nr > self.label_n_nr:
                    self.label = self.label_y
                else:
                    self.label = self.label_n

    def predict(self, f_name, input_data, row):  # Predict the Label
        if self.line_number == self.depth or self.label_y_nr == 0 or self.label_n_nr == 0:
            if row == 1:
                f = open(f_name, 'w')
            else:
                f = open(f_name, 'a')
            f.write(self.label + '\n')
            f.close()
        else:
            if input_data[self.node_nr] == self.attr_val_0[self.node_nr]:
                self.left.predict(f_name, input_data, row)
            else:
                self.right.predict(f_name, input_data, row)

    def error(self, f_name_1, f_name_2, input_data):  # Error of the Algorithm
        right = 0
        wrong = 0
        f_1 = open(f_name_1, 'r+').read().splitlines()

        for i in range(0, len(f_1)):
            if f_1[i] == input_data[i+1][-1]:
                right += 1
            else:
                wrong += 1

        err = float(wrong)/(right+wrong)

        f_2 = open(f_name_2, "a")
        f_2.write("error: " + str(err) + '\n')
        f_2.close()

n = DecisionTree()
n.generate_attr_val()
n.generate_stump()
n.generate_tree()
for t in range(0, len(tree)):
    n.learned_tree += tree[t]
print n.learned_tree

for r in range(1, len(db)):
    n.predict(label_1, db[r], r)
for t in range(1, len(db_t)):
    n.predict(label_2, db_t[t], t)

open(metrics, 'w').close()
n.error(label_1, metrics, db)
n.error(label_2, metrics, db_t)
