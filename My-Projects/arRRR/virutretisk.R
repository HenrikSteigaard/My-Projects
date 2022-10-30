#Henter data
data = "http://www.uio.no/studier/emner/matnat/math/STK1000/data + 
/obligdata/oblig1/vitruvisk.txt"
vitruvisk <- read.table(data,header=TRUE)
#Får en oppsummering av dataene:
summary(vitruvisk)
#Regner ut hvor mange menn og kvinner som deltok:
table(vitruvisk$kjonn)
#Plotter x- og y-verdiene for plottet etter oppgavespesifikasjonen:

#Bruker koorelasjonsfunksjon:
cor(vitruvisk$fot.navle, vitruvisk$kroppslengde)
plot(vitruvisk$fot.navle, vitruvisk$kroppslengde, col = 'blue', 
     xlab = "Avstand navle til gulv", ylab = "Kroppslengde")
#Legger til nødvendig tilleggskode
fit <- lm(vitruvisk$kroppslengde ~ vitruvisk$fot.navle)
abline(fit)
summary(fit)
#Skriver inn oppgitt kode i oppgaven:
plot(vitruvisk$fot.navle,residuals(fit))
abline(0,0)

#Standardiserer dataene for aa se dataene i forhold til standardaavvik.
navle.lm = lm(vitruvisk$kroppslengde ~ vitruvisk$fot.navle, data=vitruvisk) 
navle.stdres = rstandard(navle.lm)
plot(vitruvisk$fot.navle, navle.stdres)
abline(0,0)



