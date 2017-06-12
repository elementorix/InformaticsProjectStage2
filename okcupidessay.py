import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

ok_list = []
age_list = []
age_class_list = []
edu_list = []
positivity = []
negativity = []
neutrality = []
compound = []
particularitylist = []

with open('profiles_py.csv') as f:  # transforming the large data set
    file = csv.reader(f)
    next(file)
    for line in file:
        # Step 1: Aggregate state of mind!
        # Step 1.1: VADER sentiment analysis
        intro_sent = analyser.polarity_scores(line[27])  # introduction
        # job, done to see how one feels about day-to-day life
        job_sent = analyser.polarity_scores(line[28])
        skills_sent = analyser.polarity_scores(
            line[29])  # how one considers their talents
        posfeat_sent = analyser.polarity_scores(line[30])  # positive features
        thoughts_sent = analyser.polarity_scores(
            line[33])  # how one considers thoughts
        emb_sent = analyser.polarity_scores(
            line[35])  # embarssaing moments essay

        # Step 1.2: Weights and creating a list
        sent_weights = [2, 5, 1, 6, 4, 3]  # order of importance
        pos_list = [intro_sent['pos'], job_sent['pos'], skills_sent['pos'], posfeat_sent['pos'], thoughts_sent['pos'],
                    emb_sent['pos']]
        neg_list = [intro_sent['neg'], job_sent['neg'], skills_sent['neg'], posfeat_sent['neg'], thoughts_sent['neg'],
                    emb_sent['neg']]
        neu_list = [intro_sent['neu'], job_sent['neu'], skills_sent['neu'], posfeat_sent['neu'], thoughts_sent['neu'],
                    emb_sent['neu']]

        compound_list = [intro_sent['compound'], job_sent['compound'], skills_sent['compound'], posfeat_sent['compound'], thoughts_sent['compound'],
                         emb_sent['compound']]
        # Step 1.3: use numpy average function to computer with weights

        agg_pos = np.average(pos_list, weights=sent_weights)
        agg_neg = np.average(neg_list, weights=sent_weights)
        agg_neu = np.average(neu_list, weights=sent_weights)
        agg_compound = np.average(compound_list, weights=sent_weights)

        # Step 2: Particularity Calculation
        # Step 2.1: Sentiment analysis of partner essay
        partner_pref_sent = analyser.polarity_scores(line[36])
        # (Look how not-neutral someones essay about their partner is)
        partner_pref_particularity = 1 - partner_pref_sent['neu']
        # Step 2.2: Star Sign seriousness
        sign_srs = line[23]
        sign_particularity = 0
        if sign_srs == "" or sign_srs == "it doesn't matter":
            sign_particularity = 0
        elif sign_srs == "it's fun to think about":
            sign_particularity = 0.5
        elif sign_particularity == "it matters a lot":
            sign_particularity = 1
        # Step 2.3: Income seriousness
        income = float(line[10])
        income_particualrity = income / 1000000  # Max income category
        # Step 2.4: Ethnicity Seriousness
        ethnicity = line[7]
        ethnicity_particularity = 0
        for i in range(0, 5):
            if ethnicity.count('&') == i:
                ethnicity_particularity = (5 - i) / 5
        if ethnicity.count('&') > 4:
            ethnicity_particularity = 0
        # Step 2.5: Religion Seriousness
        rlgn_srs = line[20]
        rlgn_particularity = 0
        if rlgn_srs == "not too serious about it":
            rlgn_particularity = 1 / 3
        elif rlgn_srs == "somewhat serious about it":
            rlgn_particularity = 2 / 3
        elif rlgn_srs == "very serious about it":
            rlgn_particularity == 1
        # Step 2.6 Offspring Seriousnes
        off_srs = line[15]

        off_particularity = 0
        if off_srs == "might want more" or off_srs == "might want them":
            off_particularity = 0.5
        elif off_srs == "wants more" or off_srs == "wants them" or off_srs == "doesn't want any" or off_srs == "doesn't want more":
            off_particularity = 1
        # Step 2.7 Weights and Datalist
        particularity_weights = [6, 5, 4, 3, 2, 1]
        particulary_data = [partner_pref_particularity, sign_particularity,
                            income_particualrity, ethnicity_particularity, rlgn_particularity, off_particularity]
        # Step 2.8 Numpy average
        particularity = np.average(
            particulary_data, weights=particularity_weights)
        # Step 3: Combine
        ok_id = line[0]
        age = int(line[1])
        education = line[6]
        ed_class = "Primary"
        if education == "dropped out of high school" or education == "working on high school" or education == "":
            ed_class = "Primary"
        elif ("dropped out" in education or education == "high school") and education != "dropped out of high school":
            ed_class = "Secondary"
        elif "college" or "space camp" in education and "dropped out" not in education:
            ed_class = "Tertiary"
        elif "ph.d program" in education and "dropped out" not in education:
            ed_class = "Doctorate"
        else:
            ed_class = "Postgraduate"
        age_class = "18-21"
        if age <= 21:
        	age_class = "18-21"
        elif age> 21 and age <= 27:
        	age_class = "22-27"
        elif age> 27 and age<= 30:
        	age_class = "28-30"
        elif age > 30 and age<= 40:
        	age_class = "31-40"
        elif age> 40 and age<= 50:
        	age_class = "41-50"
        elif age> 50 and age <= 60:
        	age_class = "51-60"
        elif age>60:
        	age_class = "over 60"
        ok_list.append(ok_id)
        age_list.append(age)
        age_class_list.append(age_class)
        edu_list.append(ed_class)
        particularitylist.append(particularity)
        positivity.append(agg_pos)
        negativity.append(agg_neg)
        neutrality.append(agg_neu)
        compound.append(agg_compound)

DataSet = list(zip(ok_list, age_class_list, age_list, edu_list, particularitylist,
                   positivity, negativity, neutrality, compound))
df = pd.DataFrame(data=DataSet, columns=[
                  'ID', 'Age_class', 'Age', 'Education', 'Particularity', 'Positivity', 'Negativity', 'Neutrality', 'Compound'])
df.to_csv('ok_sentimentsAge.csv', sep='\t', encoding='utf-8')
