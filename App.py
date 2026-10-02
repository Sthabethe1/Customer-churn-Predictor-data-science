{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "73d63328-8f08-4666-a96f-c89d3ac1f0cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#packages \n",
    "\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f0b134ce-2e23-498f-a995-bcd6f0c84da3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-10-02 21:56:14.831 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Set up page config\n",
    "st.set_page_config(\n",
    "    page_title=\"Customer Churn Predictor\", page_icon=\"🔮\", layout=\"centered\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9e96544b-ae08-4b9e-905d-86e270ecd04d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-10-02 21:56:14.842 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:17.923 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.620 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\User\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-10-02 21:56:18.621 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.622 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.623 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.624 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.625 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.625 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.627 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.628 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load model and scaler\n",
    "@st.cache_resource\n",
    "def load_assets():\n",
    "    model = joblib.load(\"rf_model.pkl\")\n",
    "    scaler = joblib.load(\"scaler.pkl\")\n",
    "    return model, scaler\n",
    "\n",
    "\n",
    "model, scaler = load_assets()\n",
    "\n",
    "# UI Header\n",
    "st.title(\"📊 Customer Churn Prediction App\")\n",
    "st.write(\n",
    "    \"Enter the customer details below to predict the likelihood of churn.\"\n",
    ")\n",
    "\n",
    "st.markdown(\"---\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ca35cd47-816d-4f8f-aea3-7668608073b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-10-02 21:56:18.647 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.648 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.649 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.650 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.651 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.652 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.653 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.655 Session state does not function when running a script without `streamlit run`\n",
      "2026-10-02 21:56:18.656 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.657 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.659 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.660 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.662 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.664 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.665 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.666 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.666 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.667 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.668 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.669 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.669 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.670 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.671 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.672 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.672 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.673 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.674 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.675 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.679 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.681 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.681 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.682 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# User Input Form\n",
    "st.subheader(\"Customer Information\")\n",
    "\n",
    "age = st.number_input(\"Age\", min_value=18, max_value=100, value=40, step=1)\n",
    "gender = st.selectbox(\"Gender\", options=[\"Male\", \"Female\"])\n",
    "tenure = st.number_input(\n",
    "    \"Tenure (Months)\", min_value=0, max_value=120, value=12, step=1\n",
    ")\n",
    "monthly_charges = st.number_input(\n",
    "    \"Monthly Charges ($)\", min_value=0.0, max_value=500.0, value=70.0, step=1.0\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "507168ea-b0f3-470a-a669-6ed63ce30440",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encode Gender (Matching training encoding: Female = 1, Male = 0)\n",
    "gender_encoded = 1 if gender == \"Female\" else 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d981a912-50d6-4b1a-be03-b035059b02a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-10-02 21:56:18.714 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.715 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.716 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.717 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.718 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.719 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.722 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.732 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-10-02 21:56:18.754 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Predict Button\n",
    "st.markdown(\"---\")\n",
    "if st.button(\"Predict Churn Status\", type=\"primary\"):\n",
    "    # Create input DataFrame matching exact model features\n",
    "    input_data = pd.DataFrame(\n",
    "        [[age, gender_encoded, tenure, monthly_charges]],\n",
    "        columns=[\"Age\", \"Gender\", \"Tenure\", \"MonthlyCharges\"],\n",
    "    )\n",
    "\n",
    "    # Scale inputs using saved scaler\n",
    "    input_scaled = scaler.transform(input_data)\n",
    "\n",
    "    # Make prediction and probabilities\n",
    "    prediction = model.predict(input_scaled)[0]\n",
    "    probability = model.predict_proba(input_scaled)[0][1]\n",
    "\n",
    "    # Display Results\n",
    "    st.subheader(\"Prediction Result\")\n",
    "\n",
    "    if prediction == 1:\n",
    "        st.error(\n",
    "            f\"⚠️️ **High Risk of Churn!** (Probability: {probability * 100:.1f}%)\"\n",
    "        )\n",
    "    else:\n",
    "        st.success(\n",
    "            f\"✅ **Low Risk / Retention Likely** (Churn Probability: {probability * 100:.1f}%)\"\n",
    "        )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
