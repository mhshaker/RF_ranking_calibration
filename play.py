import scikit_posthocs as sp

x = [[1,2,3,4,5,6,7,8,9], [10,20,30,40,50,60,70,80,90]]
print("ttest", sp.posthoc_ttest(x, p_adjust = 'holm')) 
