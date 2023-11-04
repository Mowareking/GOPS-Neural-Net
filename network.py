import random
import numpy as np
import json
import sys
from data_loader import load_gops

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda, test_data=None):
        
        training_data = list(training_data)
        num_training = len(training_data)

        if test_data:
            test_data = list(test_data)
            num_test = len(test_data)
            best_accuracy = 0

        for epoch in range(1, epochs+1):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, num_training, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, num_training)
            if test_data:
                accuracy = self.evaluate(test_data)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save("best_network.json")
                print("Epoch {0}: {1} / {2}".format(epoch, accuracy, num_test))
            else:
                print ("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def load(filename):
        """Load a neural network from the file ``filename``.  Returns an
        instance of Network.

        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        net = Network(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def sum_cards(cards):
    result = 0
    for card in cards:
        if card == "A": 
            result += 1
        elif card == "J":
            result += 11
        elif card == "Q":
            result += 12
        elif card == "K":
            result += 13
        else:
            result += int(card)
    return result

def versus(net):
    computer_hand = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] 
    opponent_hand = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] 
    stock = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] 
    random.shuffle(stock)
    upturned = []
    opponent_prizes = []
    computer_prizes = []
    for i in range(13):
        upturned.append(stock.pop())
        data = []
        for cards in [computer_hand, opponent_hand, stock]:
            if "A" in cards:
                data.append(1)
            else:
                data.append(0)
            for i in range(2, 11):
                if str(i) in cards:
                    data.append(1)
                else:
                    data.append(0)
            for royal in ["J", "Q", "K"]:
                if royal in cards:
                    data.append(1)
                else:
                    data.append(0)
        data.append(int(sum_cards(upturned)))
        data = np.reshape(np.array(data), (40, 1))

        dist = net.feedforward(data)
        while True:
            try:
                computer_move = np.argmax(dist) + 1
            except ValueError:
                computer_move = random.choice(computer_hand)
                if computer_move == "A":
                    computer_move = 1
                elif computer_move == "J":
                    computer_move = 11
                elif computer_move == "Q":
                    computer_move = 12
                elif computer_move == "K":
                    computer_move = 13
                else:
                    computer_move = int(computer_move)
            if computer_move == 1:
                computer_move_display = "A"
            elif computer_move == 11:
                computer_move_display = "J"
            elif computer_move == 12:
                computer_move_display = "Q"
            elif computer_move == 13:
                computer_move_display = ("K")
            else:
                computer_move_display = str(computer_move)
            if computer_move_display in computer_hand:
                computer_hand.remove(computer_move_display)
                break
            dist = np.delete(dist, computer_move-1)

        print(upturned)
        print(f"Your hand: {opponent_hand}")
        print(f"Your prizes: {opponent_prizes}")
        print(f"Computer prizes: {computer_prizes}")
        opponent_move = input("Enter your move: ")
        opponent_hand.remove(opponent_move)
        
        print(f"\nYou played {opponent_move}. Computer played {computer_move_display}.")
        if opponent_move == "A":
            opponent_move = 1
        elif opponent_move == "J":
            opponent_move = 11
        elif opponent_move == "Q":
            opponent_move = 12
        elif opponent_move == "K":
            opponent_move = 13
        else:
            opponent_move = int(opponent_move)
        
        if computer_move > opponent_move:
            print("Computer won the card.\n")
            computer_prizes.extend(upturned)
            upturned = []
        elif computer_move < opponent_move:
            print("You won the card.\n")
            opponent_prizes.extend(upturned)
            upturned = []
        else:
            print("It's a tie.\n")
    print(f"Final scores:\nYour prize total: {sum_cards(opponent_prizes)} Computer prize total: {sum_cards(computer_prizes)}")
    

net = Network.load("best_network.json")
versus(net)
#tr, te = load_gops()
#net.SGD(tr, 70, 10, 0.5, 0.1, te)