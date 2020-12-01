BMI = ['body_mass_index']

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
              'aspartate_transaminase'
              'lactate_dehydrogenase',
              'albumin',
              'creatine_kinase',
              'gamma_glutamyl_transferase',
              'alkaline_phosphatase',
              'bilirubin_direct',
              'bilirubin_total']

BLOOD_GAS = ['pco2_arterial',
             'po2_arterial',
             'ph_arterial',
             'bicarbonate_arterial',
             'lactate_arterial' ]

CENTRAL_LINE = ['so2_central_venous']

VITAL_SIGNS = ['heart_rate',
               'arterial_blood_pressure_mean',
               'arterial_blood_pressure_diastolic',
               'arterial_blood_pressure_systolic']

VENTILATOR_VALUES = ['peep',
                     'fio2',
                     'pressure_above_peep',
                     'tidal_volume',
                     'inspiratory_expiratory_ratio',
                     'respiratory_rate_measured',
                     'respiratory_rate_set',
                     'respiratory_rate_measured_ventilator'
                     'lung_compliance',
                     'driving_pressure',
                     'plateau_pressure',
                     'peak_pressure']

def get_covariates(df, covariates = None):

    if not covariates:
        covariates = BMI + LAB_VALUES + BLOOD_GAS + CENTRAL_LINE + VITAL_SIGNS + VENTILATOR_VALUES


