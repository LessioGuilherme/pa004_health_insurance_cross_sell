import json
import pickle
import pandas as pd

class InsuranceVehicle(object):
    def __init__( self ):
        self.home_path = 'C:/Users/Guilherme/Repos/pa004_health_insurance_cross_sell/'
        self.ss_annual_premium                   = pickle.load ( open(self.home_path +'/parameter/ss_annual_premium_scaler.pkl', 'rb'))
        self.ss_annual_premium_per_day           = pickle.load ( open(self.home_path +'/parameter/ss_annual_premium_per_day_scaler.pkl', 'rb'))
        self.ss_annual_premium_per_age           = pickle.load ( open(self.home_path +'/parameter/ss_annual_premium_per_age_scaler.pkl', 'rb'))
        self.ss_annual_premium_per_ap_mean_rc    = pickle.load ( open(self.home_path +'/parameter/ss_annual_premium_per_ap_mean_rc_scaler.pkl', 'rb'))
        self.ss_annual_premium_per_ap_mean_psc   = pickle.load ( open(self.home_path +'/parameter/ss_annual_premium_per_ap_mean_psc_scaler.pkl', 'rb'))
        self.mms_vintage                         = pickle.load ( open(self.home_path +'/parameter/mms_vintage_scaler.pkl', 'rb'))
        self.mms_age                             = pickle.load ( open(self.home_path +'/parameter/mms_age_scaler.pkl', 'rb'))
        self.mms_vintage_per_age                 = pickle.load ( open(self.home_path +'/parameter/mms_vintage_per_age_scaler.pkl', 'rb'))
        self.mms_vehicle_damage_mean_region_code = pickle.load ( open(self.home_path +'/parameter/mms_vehicle_damage_mean_region_code_scaler.pkl', 'rb'))
        
    def data_cleaning (self, df1):
        df1.columns = ['id', 'gender', 'age', 'driving_license', 'region_code',
                           'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
                           'policy_sales_channel', 'vintage']
        df1['region_code'] = df1['region_code'].astype(np.int64)
        df1['policy_sales_channel'] = df1['policy_sales_channel'].astype(np.int64)
        
        return df1
    
    def feature_engineering (self, df2):
        #Creating new features
        
        #mapping vechicle_damage from NO to 0 and Yes to 1
        vehicle_damage_mapping = {'No':0,
                                  'Yes':1}
        df2['vehicle_damage'] = df2['vehicle_damage'].map(vehicle_damage_mapping)

        #annual_premium paid per day
        df2['annual_premium_per_day'] = df2['annual_premium']/df2['vintage']

        #annual_premium divided per age
        df2['annual_premium_per_age'] = df2['annual_premium']/df2['age']

        #vintage_per_age
        df2['vintage_per_age'] = df2['vintage']/df2['age']

        #logic between previously_insured  and vehicle_damage
        df2['previously_insured_vehicle_damage'] =  df2.apply(lambda row: -(row['vehicle_damage'] + row['previously_insured'])**2 if row['previously_insured'] == 0 else (row['vehicle_damage'] + row['previously_insured'])**2, axis=1)

        #logic between vehicle_age (consider age > 1 as 1 and age < 1 as 0) and vehicle_damage
        df2['vehicle_age_vehicle_damage'] = df2.apply(lambda row: -((row['vehicle_damage'] + 0)**2) if row['vehicle_age'] == '<1 Year' else (row['vehicle_damage'] + 1)**2, axis=1)

        #logic between driving_license and vehicle_damage
        df2['vehicle_damage_license'] = df2.apply(lambda row: -(row['vehicle_damage'] + row['driving_license'])**2 if row['driving_license'] == 0 else (row['vehicle_damage'] + row['driving_license'])**2, axis=1)

        #logic between annual_premium divided per mean of annual_premium per region_code
        df2_annual_premium_mean_region_code = df2.rename(columns={'annual_premium': 'annual_premium_mean'}).groupby('region_code').mean()
        df2_annual_premium_mean_region_code.reset_index(inplace=True)
        df2 = df2.merge(df2_annual_premium_mean_region_code[['region_code', 'annual_premium_mean']], on='region_code', how='left')
        df2['annual_premium_per_ap_mean_rc'] = df2.apply(lambda row: (row['annual_premium']/row['annual_premium_mean']) ,axis=1)
        df2.drop(columns='annual_premium_mean', inplace = True)
        df2_annual_premium_mean_region_code = None

        #logic between annual_premium divided per mean of annual_premium per policy_sales_channel
        df2_annual_premium_mean_policy_sales_channel = df2.rename(columns={'annual_premium': 'annual_premium_mean'}).groupby('policy_sales_channel').mean()
        df2_annual_premium_mean_policy_sales_channel.reset_index(inplace=True)
        df2 = df2.merge(df2_annual_premium_mean_policy_sales_channel[['policy_sales_channel', 'annual_premium_mean']], on='policy_sales_channel', how='left')
        df2['annual_premium_per_ap_mean_psc'] = df2.apply(lambda row: (row['annual_premium']/row['annual_premium_mean']) ,axis=1)
        df2.drop(columns='annual_premium_mean', inplace = True)
        df2_annual_premium_mean_policy_sales_channel = None

        #logic of vehicle_damage_mean per region_code
        df2_vehicle_damage_mean_per_region_code = df2.rename(columns={'vehicle_damage': 'vehicle_damage_mean_region_code'}).groupby('region_code').mean()
        df2_vehicle_damage_mean_per_region_code.reset_index(inplace=True)
        df2 = df2.merge(df2_vehicle_damage_mean_per_region_code[['region_code', 'vehicle_damage_mean_region_code']], on='region_code', how='left')
        
        return df2

    def data_preparation (self, df4):
        
        # Standarlization
        # annual_premium
        df4['annual_premium'] = self.ss_annual_premium.fit_transform(df4[['annual_premium']].values)
        # annual_premium_per_day
        df4['annual_premium_per_day'] = self.ss_annual_premium_per_day.fit_transform(df4[['annual_premium_per_day']].values)
        # annual_premium_per_age
        df4['annual_premium_per_age'] = self.ss_annual_premium_per_age.fit_transform(df4[['annual_premium_per_age']].values)
        # annual_premium_per_ap_mean_rc
        df4['annual_premium_per_ap_mean_rc'] = self.ss_annual_premium_per_ap_mean_rc.fit_transform(df4[['annual_premium_per_ap_mean_rc']].values)
        # annual_premium_per_ap_mean_psc
        df4['annual_premium_per_ap_mean_psc'] = self.ss_annual_premium_per_ap_mean_psc.fit_transform(df4[['annual_premium_per_ap_mean_psc']].values)

        # Rescaling
        # vintage
        df4['vintage'] = self.mms_vintage.fit_transform(df4[['vintage']].values)
        # age
        df4['age'] = self.mms_age.fit_transform(df4[['age']].values)
        # vintage_per_age
        df4['vintage_per_age'] = self.mms_vintage_per_age.fit_transform(df4[['vintage_per_age']].values)
        # vehicle_damage_mean_region_code
        df4['vehicle_damage_mean_region_code'] = self.mms_vehicle_damage_mean_region_code.fit_transform(df4[['vehicle_damage_mean_region_code']].values)

        columns_selected = ['age', 
                            'vintage_per_age', 
                            'annual_premium_per_day',
                            'vintage',    
                            'annual_premium_per_age',   
                            'annual_premium_per_ap_mean_rc',    
                            'annual_premium_per_ap_mean_psc',    
                            'annual_premium',
                            'previously_insured',
                            'vehicle_damage',
                            'vehicle_damage_mean_region_code'
                            ]
        return df4[columns_selected]
    
    def get_ranking(self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba(test_data)

        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        original_data=original_data.sort_values('score',ascending=False)
        original_data=original_data.reset_index(drop=True)
        original_data['ranking']=original_data.index+1
        
        return original_data.to_json(orient='records', date_format='iso')