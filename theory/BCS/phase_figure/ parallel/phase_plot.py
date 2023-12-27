
#max_qのデータを読み取り






# q=0をBCSへq＝＼0をFFLOへと色分けする
for h in range(n2):
    for i in range(n1):
            figure = plt.scatter(kBTs[h], Bs[i], 5, c = q_max_1[h][i], cmap='viridis' ,vmin=0, vmax=n0) # c = q_max[h][i]
c= plt.colorbar()
plt.savefig("FFLO_BCS_phase_figure.png")
plt.show()



