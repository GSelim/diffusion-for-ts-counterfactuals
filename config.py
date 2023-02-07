
dataset="GunPoint"

ts_size =148
timesteps=100
channels = 1
batch_size = 15
convnext=False
epochs=21

#à l'origine était à 8
resnetBlockGroups=2


#fixer à True si vous souhaiter utiliser la normalisation
#je en suis pas sur qu'elle soit correctement mise en place, elle ne semble pas améliorer l'apprentissage
normalisation_test=False

#cosine linear quadratic sigmoid
schedule="linear"


#in residual
ConvKernelSize=4
ConvStrideSize=2
ConvPaddingSize=1


#in block
blockConvKernelSize=3
blockConvPadding=1