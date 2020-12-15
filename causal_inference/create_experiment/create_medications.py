MEDICATIONS = ['med_anesthetics',
               'med_psycholeptics',
               'med_muscle_relaxants',
               'med_cardiac_stimulants_excl_cardiac_glycosides',
               'med_opioids',
               'med_vitamin_k_antagonists',
               'med_heparin_group',
               'med_direct_thrombin_inhibitors',
               'med_direct_factor_xa_inhibitors',
               'med_other_antithrombotic_agents',
               'med_high-ceiling_diuretics',
               'med_expectorants_excl_combinations_with_cough_suppressants',
               'med_adrenergics_inhalants',
               'med_other_drugs_for_obstructive_airway_diseases_inhalants',
               'med_norepinephrine',
               'med_dopamine',
               'med_epinephrine',
               'med_vasopressin_(argipressin)',
               'med_terlipressin',
               'med_corticosteroids_for_systemic_use_plain']


def get_medications(dl):

    df_medications = dl.get_medications(columns=['hash_patient_id',
                                                 'pacmed_name',
                                                 'pacmed_subname',
                                                 'start_timestamp',
                                                 'end_timestamp',
                                                 'total_dose',
                                                 'dose_unit_name'])

    medications = (df_medications['pacmed_name'].isin(MEDICATIONS)) | \
                  (df_medications['pacmed_subname'].isin(MEDICATIONS))

    df_medications = df_medications.loc[medications]

    return df_medications
