


########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(kBTs, xi, 5)
plt.savefig("figure/q-DeltaF.png")
plt.clf()

###################################
##output
file = open("output/q-\DeltaF", "w")
file.write("##q-\DeltaF" + "\n")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()


