BMI = ['body_mass_index']

SOFA = ['sofa_score']

LAB_VALUES = ['c_reactive_protein',
              'leukocytes',
              'procalcitonin',
              'hemoglobin',
              'creatinine',
              'ureum',
              'sodium',
              'potassium',
              'calcium',
              'phosphate',
              'chloride',
              'glucose',
              'activated_partial_thromboplastin_time',
              'prothrombin_time_inr',
              'prothrombin_time_sec',
              'd_dimer',
              'fibrinogen',
              'alanine_transaminase',
              'aspartate_transaminase',
              'lactate_dehydrogenase',
              'albumin',
              'creatine_kinase',
              'gamma_glutamyl_transferase',
              'alkaline_phosphatase',
              'bilirubin_direct',
              'bilirubin_total']

BLOOD_GAS = ['pco2_arterial',
             'pco2_unspecified',
             'po2_arterial',
             'po2_unspecified',
             'bicarbonate_arterial',
             'bicarbonate_unspecified',
             'ph_arterial',
             'ph_unspecified',
             'ph_central_venous',
             'ph_mixed_venous',
             'ph_venous',
             'lactate_arterial',
             'lactate_blood',
             'lactate_unspecified',
             'lactate_mixed_venous',
             'lactate_venous'
             ]


CENTRAL_LINE = ['so2_central_venous']

SATURATION = ['o2_saturation']

VITAL_SIGNS = ['heart_rate',
               'arterial_blood_pressure_mean',
               'arterial_blood_pressure_diastolic',
               'arterial_blood_pressure_systolic']

VENTILATOR_VALUES = ['peep',
                     'fio2',
                     'pressure_above_peep',
                     'tidal_volume',
                     'tidal_volume_per_kg',
                     'tidal_volume_per_kg_set',
                     'inspiratory_expiratory_ratio',
                     'respiratory_rate_measured',
                     'respiratory_rate_set',
                     'respiratory_rate_measured_ventilator',
                     'lung_compliance_dynamic',
                     'lung_compliance_static',
                     'driving_pressure',
                     'plateau_pressure',
                     'peak_pressure']

CVVH = ['cvvh_blood_flow', 'cvvhd_blood_flow', 'aki']

COVARIATES_8h = BLOOD_GAS + CENTRAL_LINE + SATURATION + VITAL_SIGNS + VENTILATOR_VALUES + CVVH