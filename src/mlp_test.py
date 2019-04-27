import mlp_xor as mymlp

mlp1 = mymlp.MLP2Neuron()

mlp2 = mymlp.MLP4Neuron()

mlp3 = mymlp.MLP8Neuron()

Y_64 = mymlp.np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]])

test_64 = X_64 = mymlp.np.array([[-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_0(), mymlp.generate_noise_for_1()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_0()],
                                 [-1, mymlp.generate_noise_for_1(), mymlp.generate_noise_for_1()]])

''' Convergence test'''
mlp1.run_16()

mlp2.run_16()

mlp3.run_16()

mymlp.plt.legend()
mymlp.plt.show()

''' Average losses test '''
Y1 = mlp1.forward(test_64)

Y2 = mlp2.forward(test_64)

Y3 = mlp3.forward(test_64)

Y1_train = mlp1.forward(mymlp.X_64)

Y2_train = mlp2.forward(mymlp.X_64)

Y3_train = mlp3.forward(mymlp.X_64)

loss1 = mymlp.loss(Y_64.T, Y1)

loss2 = mymlp.loss(Y_64.T, Y2)

loss3 = mymlp.loss(Y_64.T, Y3)

loss1_train = mymlp.loss(mymlp.Y_64.T, Y1_train)

loss2_train = mymlp.loss(mymlp.Y_64.T, Y2_train)

loss3_train = mymlp.loss(mymlp.Y_64.T, Y3_train)

print ("======== Test Results ========")

print ("2 units: Test data: " + str(loss1) + " Training data: " + str(loss1_train))

print ("4 units: Test data: " + str(loss2) + " Training data: " + str(loss2_train))

print ("8 units: Test data: " + str(loss3) + " Training data: " + str(loss3_train))

''' Generalisation performance test'''
mlp1.generalisation_test()
mymlp.plt.legend()
mymlp.plt.show()

mlp2.generalisation_test()
mymlp.plt.legend()
mymlp.plt.show()

mlp3.generalisation_test()
mymlp.plt.legend()
mymlp.plt.show()

''' Mapping function visualization'''
mlp1.draw_network()
mymlp.plt.show()

mlp2.draw_network()
mymlp.plt.show()

mlp3.draw_network()
mymlp.plt.show()

