# IMPORTING ALL NECESSARY LIBRARIES
import re
import numpy as np
import pandas as pd

# Switching off the pandas warnings
pd.set_option('mode.chained_assignment', None)

# TASK 1 : DATA DESCRIPTION

# Loading dataset filepath and reading csv file
data = 'C:\\Users\\Jenit\\Desktop\\resume_data.csv'
dataset = pd.read_csv(data, encoding='ISO-8859-1')

# Getting number of rows and columns in the dataset
datasize = dataset.shape
print(datasize)

# Getting datatypes of the dataset attributes (columns)
column_datatype = dataset.dtypes
print(column_datatype)

# Getting concise summary of the dataframe
dataset_summary = dataset.info()
print(dataset_summary)

# Making index column as Job Applicant ID
dataset.reset_index(level=0, inplace=True)
dataset = dataset.rename(columns={'index': 'Job_Applicant_ID'})
dataset['Job_Applicant_ID'] = (dataset['Job_Applicant_ID'] + 1)

# TASK 2 : DATA CLEANING

# Check for missing data
print(dataset.isnull().any().any())

# Check for unreasonable data
print(dataset.applymap(np.isreal))

# Count the missing values in each column of the dataframe
print(dataset.isnull().sum())

# Dropping all rows with null values
process_dataset = dataset.dropna(how='any')

# Check for missing data after dropping operation to confirm non-null dataset
print(process_dataset.isnull().any().any())


# TASK 3 : RESUME PARSING AND FEATURE EXTRACTION (BY REGULAR EXPRESSION APPROACH)

# Function definition to parse each CV record in dataset and remove unwanted keywords or symbols from the dataset
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+-./:;<=>?@[\]^_`{|}~"""), ' ',
                         resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub("\s+", " ", resume_text)  # remove extra whitespace
    return resume_text


# Applying the clean_resume function to each attribute in dataset
process_dataset['Resume_title'] = process_dataset.Resume_title.apply(lambda x: clean_resume(x))
process_dataset['Description'] = process_dataset.Description.apply(lambda x: clean_resume(x))
process_dataset['work_experiences'] = process_dataset.work_experiences.apply(lambda x: clean_resume(x))
process_dataset['Educations'] = process_dataset.Educations.apply(lambda x: clean_resume(x))
process_dataset['Skills'] = process_dataset.Skills.apply(lambda x: clean_resume(x))
process_dataset['Certificates'] = process_dataset.Certificates.apply(lambda x: clean_resume(x))

# Saving the parse dataset with required information in below mention path of the system
process_dataset.to_csv('C:\\Users\\Jenit\\Desktop\\clean_resume_data.csv', index=False)

# Replacing few values such as 'none' to '0' and 'present' to '2020' in dataset for meaningful information
job_category = pd.read_csv('C:\\Users\\Jenit\\Desktop\\clean_resume_data.csv', encoding='ISO-8859-1')
job_category = job_category.replace(regex=['Present'], value='2020')
job_category['work_experiences'] = job_category['work_experiences'].replace(regex=['NONE'], value='0')
job_category['Skills'] = job_category['Skills'].replace(regex=['NONE'], value='0')

# Printing on Console Window for user friendly GUI
print("\n\nPERSONALITY PREDICTION SYSTEM THROUGH CV ANALYSIS \n\n")
print(
    "---GENERAL GUIDELINES---\n\nFor Job Post as an user input, you can simply type ----\n 'Java' for Java Developer "
    "\n 'Python' for Python Developer \n 'PHP' for PHP Developer \n and so on...\n\n")

# Allowing user to input the job post to screen CVs
job_requirement = input("Enter the Job Post for CV Screening : ")
job_requirement_specification = "0 wtitle " + job_requirement.capitalize() + " Developer "

# The 'if' condition to check if there exists any job applicant that user wants to recruit for specific job post
if len(job_category[job_category['work_experiences'].str.contains(job_requirement_specification)]) > 0:

    job_category_records = job_category[job_category['work_experiences'].str.contains(job_requirement_specification)]
    job_category_records.to_csv('C:\\Users\\Jenit\\Desktop\\job_applicants.csv', index=False)

    job_post_applicants = pd.read_csv('C:\\Users\\Jenit\\Desktop\\job_applicants.csv', encoding='ISO-8859-1')

    # Extracting work experience feature from the dataset
    for job_index_1, job_experience_years in job_post_applicants.iterrows():
        job_keyword_match = job_experience_years['work_experiences'].split(',')
        if re.findall('(\d{4})', job_keyword_match[4]):
            job_years = re.findall('(\d{4})', job_keyword_match[4])
            if len(job_years) == 2:
                job_post_applicants.loc[job_index_1, 'Work_Experience_Duration'] = (int(job_years[1])) - (
                    int(job_years[0]))
            else:
                job_post_applicants.loc[job_index_1, 'Work_Experience_Duration'] = 0.5
        else:
            job_post_applicants.loc[job_index_1, 'Work_Experience_Duration'] = 0

    # Extracting skills feature from the dataset
    for job_index_2, job_skills_counter in job_post_applicants.iterrows():
        job_no_of_skills = job_skills_counter['Skills'].split(',')
        job_post_applicants.loc[job_index_2, 'Number_of_Skills'] = (len(job_no_of_skills))

    job_post_applicants['Number_of_Skills'] = pd.to_numeric(job_post_applicants['Number_of_Skills'], downcast='integer')

    # Extracting education feature from the dataset
    for job_index_3, job_education_verification in job_post_applicants.iterrows():
        job_education_match = job_education_verification['Educations'].split(',')
        if ("Computer" in job_education_match[0]) or ("IT" in job_education_match[0]):
            job_post_applicants.loc[job_index_3, 'Education_Background_Check'] = 1
        else:
            job_post_applicants.loc[job_index_3, 'Education_Background_Check'] = 0

    job_post_applicants['Education_Background_Check'] = pd.to_numeric(job_post_applicants['Education_Background_Check'],
                                                                      downcast='integer')

    # Extracting certification feature from the dataset
    for job_index_4, job_certificate_verification in job_post_applicants.iterrows():
        job_certificate_match = job_certificate_verification['Certificates'].split(',')
        if job_requirement in job_certificate_match[0]:
            job_post_applicants.loc[job_index_4, 'Certificate_Check'] = 1
        else:
            job_post_applicants.loc[job_index_4, 'Certificate_Check'] = 0

    job_post_applicants['Certificate_Check'] = pd.to_numeric(job_post_applicants['Certificate_Check'],
                                                             downcast='integer')

    # Saving feature extracted dataset or fact table in the system which may be used by Recruiter for reference
    job_post_applicants.to_csv('C:\\Users\\Jenit\\Desktop\\job_fact_table.csv', index=False,
                               columns=['Job_Applicant_ID', 'Work_Experience_Duration', 'Number_of_Skills',
                                        'Education_Background_Check', 'Certificate_Check'])

    # TASK 4 : JOB APPLICANT CV SCORING  (BY AVERAGE WEIGHT APPROACH)

    job_applicants_score = pd.read_csv('C:\\Users\\Jenit\\Desktop\\job_fact_table.csv', encoding='ISO-8859-1')

    # Allowing user to input weightage for each criteria or feature to score job applicant's performance
    weight_job_post_work_experience = float(input("\nEnter weight assigned for work experience duration (0 to 1) : "))
    weight_job_post_skills = float(input("\nEnter weight assigned for work skills (0 to 1) : "))
    weight_job_post_education: float = float(input("\nEnter weight assigned for education (0 to 1) : "))
    weight_job_post_certificate = float(input("\nEnter weight assigned for certification (0 to 1) : "))

    # Determining maximum value in each criteria to know the highest performance that can be achieved
    max_value_experience_col = int(job_applicants_score['Work_Experience_Duration'].max())
    max_value_skills_col = int(job_applicants_score['Number_of_Skills'].max())
    max_value_education_col = int(job_applicants_score['Education_Background_Check'].max())
    max_value_certificate_col = int(job_applicants_score['Certificate_Check'].max())

    # Normalising the weight using value of each criteria and dividing it to maximum value of each criteria
    job_applicants_score['I1_WE'] = ((job_applicants_score['Work_Experience_Duration'] / max_value_experience_col)
                                     * weight_job_post_work_experience)
    job_applicants_score['I2_S'] = ((job_applicants_score['Number_of_Skills'] / max_value_skills_col)
                                    * weight_job_post_skills)
    job_applicants_score['I3_EBC'] = ((job_applicants_score['Education_Background_Check'] / max_value_education_col)
                                      * weight_job_post_education)
    job_applicants_score['I4_CC'] = ((job_applicants_score['Certificate_Check'] / max_value_certificate_col)
                                     * weight_job_post_certificate)

    # Calculating average weight by using above estimated normalised weight multiplied with values of each criteria
    # and divided by normalised weight
    job_applicants_score['Crips_Output_Per_Job_Applicant'] = (((job_applicants_score['I1_WE'] * job_applicants_score[
        'Work_Experience_Duration']) + (job_applicants_score['I2_S'] * job_applicants_score['Number_of_Skills']) + (
                                                                       job_applicants_score['I3_EBC'] *
                                                                       job_applicants_score[
                                                                           'Education_Background_Check']) + (
                                                                       job_applicants_score['I4_CC'] *
                                                                       job_applicants_score[
                                                                           'Certificate_Check'])) / (
                                                                      job_applicants_score['I1_WE'] +
                                                                      job_applicants_score['I2_S'] +
                                                                      job_applicants_score['I3_EBC'] +
                                                                      job_applicants_score['I4_CC']))

    # Saving job applicants score in the system which may be used by Recruiter for reference
    job_applicants_score_result = job_applicants_score.sort_values('Crips_Output_Per_Job_Applicant', ascending=False)
    job_applicants_score_result.to_csv('C:\\Users\\Jenit\\Desktop\\job_applicants_score.csv', index=False)

    # TASK 5 : SHORTLISTING JOB APPLICANTS

    shortlist_job_applicants = process_dataset.merge(job_applicants_score_result, on='Job_Applicant_ID')
    shortlist_job_applicants = shortlist_job_applicants.head(10)

    # Saving shortlisted job applicants in the system which may be used by Recruiter for reference
    shortlist_job_applicants.to_csv('C:\\Users\\Jenit\\Desktop\\shortlist_job_applicants.csv', index=False,
                                    columns=['Job_Applicant_ID', 'Resume_title', 'City', 'State', 'Description',
                                             'work_experiences', 'Educations', 'Skills', 'Links', 'Certificates',
                                             'Additional Information'])

    # Printing on console window about shortlisted job applicants and its location where it is stored in the system
    print("\n\nShortlisted applicants for the job post", job_requirement,
          "developer are displayed at file location --- C:\\Users\\Jenit\\Desktop\\shortlist_job_applicants.csv",
          "\n\nThank You, Have a nice day!!!")

# if there are no job applicants found in the dataset that suits the specific job post
else:
    print("\n\nThere are no job applicants found for the job post", job_requirement,
          "developer", "\nSorry,Try for another job post!!!")


