1. The data in the QR code next to the person is encoded like this:

mask,age,was_vaccinated,doctor,hours_exercise,right_vaccine

The values of the variables can be:
mask = 0 or 1
age = 0 to 100
was_vaccinated = 0 or 1
doctor = red, green, blue or yellow
hours_exercise = 0 to 40
right_vaccine = Greenzer, Rederna, StellaBluera or BlacknikV

2. The data in the QR codes on the cylinders is a link to a file on the web.
The file containes comma separated values, from which you should learn a classifier. 
There are two features, and one descrete value we want to predict.

age, hours_exercise, right_vaccine
