# team15_ClickPrediction_TCD_ML_Competition

This repo contains the submission file for team 15 in the competition https://www.kaggle.com/c/tcd-ml-comp-201920-rec-alg-click-pred-group/leaderboard

We proceded by dividing the dataset into 3 parts so that every team member can work on 1 and optimize it
The way the datasets were divided are

BB49B3789EC2F1C13237 - JabRef

B494E26228A0547AA54C - Blogs

D5036AFB177566430360 - MyVolts

The three codes corresponding to the the three files are provided here.
Blogs.py contain the code for the blogs dataset
MyVolts.py contain the code for the MyVolts dataset
JabRef.ipynb contains the code for JabRef dataset

The python files connecting documents contains the code for concatinating the outputs from the three files while separating the database contains the code of separating the database into 3 training and 3 testing files 


There are 4 output files

1. Output_JabRef_descisiontree - This file is produced from the JabRef code
2. Output_MyVolts - This file is produced form MyVolts code
3. Output_blog - This file is produced from Blogs code
4. sub_5_forst_1 - This file is produced from combining all files. Also note that the first set_clicked value (recommendatin_set_id = 46914)
 has been changed to 1 as there was a bug on kaggle because of which the public score was coming out as 0.
