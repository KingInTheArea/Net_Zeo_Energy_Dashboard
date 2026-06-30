This Repo is the codebase for the prototype of Energy Consumption Simulator for the Buildings based on various paramenters like
wall material , Roof type, Window layout, facade size etc.

In this Prototype we have used Random Forest after trying various regression techniques. since the data was very well structured as it was derived from simulator platforms like Energy Plus and one more (I dont know Exact name of platform but is similar to energy plus).

This codebase has **Verify\_accuracy** named Python script which gives R^2 value for the Model.
**Implementation:**
It trains Random forest in two passes. First it works on **Geometrical** parameters and then trains on **Material** parameters. This is due to ***Feature Dominance*** in these parameters.
Window to wall ratio and Wall and roof material are the most dominating features of this dataset in context of **Total Energy demand** and **Cooling energy demand** (Discomfort hours depend on more than one feature) and in real-life too these features dominate Energy and Cooling demand.
This information was provided by the Supervisor when giving Dataset and **validated** by a feature extraction script and it can been seen that some part of script has been kept in "*train\_models.py*" for geometrical dataset as for other user of repo can append snippet accordingly.
This ML logic has been used in another project of my team with a good frontend and UI.
This repo can be run by using "Streamlit" framework.

