# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:50:55 2017

@author: Ayush Talwar
"""


# Run Reliability Training and Outputs
print("Running reliability training and predictions")
import Reliability_Outer_Loop
import Reliability_Predictions

# Run Crew Scoring
print("Running crew scoring")
import Step_4_Crew_Scoring

# Run Cost Model based predictions
print("Running cost models training and predictions")
import Step_5_Final_Training_Outer_Loop
import Step_5_Final_Predictions

# Run circuit similarity
print("Running circuit similarity")
import Step_4_Productivity_Similarity