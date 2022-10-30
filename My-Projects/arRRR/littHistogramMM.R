#Henter data
data="http://www.uio.no/studier/emner/matnat/math/STK1000/data/obligdata + 
/oblig1/kjerneprover.txt"
kjerneprover <- read.table(data,header=TRUE)
#Får table og pie for dybde depth
table(kjerneprover$depth)
pie(table(kjerneprover$depth))


kjerneprover.dypt <- kjerneprover[kjerneprover[, "depth"]==1,]
kjerneprover.grunt <- kjerneprover[kjerneprover[,"depth"]==0,]
#Lager et histogram for dype prøver og prosentandel sand:
hist(kjerneprover.dypt$sand)
#Lager et boxplot for dype prøver og prosentandel sand:
boxplot(kjerneprover.dypt$sand)
#Lager et histogram for grunne prøver og prosentandel sand:
hist(kjerneprover.grunt$sand)
#Lager et boxplot for grunne prøver og prosentandel sand:
boxplot(kjerneprover.grunt$sand)
#Bruker median og mean for median og gjennomsnitt av grunne prøver:
median(kjerneprover.grunt$sand)
mean(kjerneprover.grunt$sand)
#Bruker median og mean for median og gjennomsnitt av dype prøver:
median(kjerneprover.dypt$sand)
mean(kjerneprover.dypt$sand)
#Bruker qqnorm og qqline for å vurdere om det er rimelig å anta
#en normalfordeling for dataene
qqnorm(kjerneprover.dypt$sand)
qqline(kjerneprover.dypt$sand)
qqnorm(kjerneprover.grunt$sand)
qqline(kjerneprover.grunt$sand)
shapiro.test(kjerneprover.dypt$sand)




