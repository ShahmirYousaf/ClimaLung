import React from "react";
import "./About.scss";
import Sidebar from "../../Components/Sidebar/Sidebar";

const About = () => {
  return (
    <div className="about-container">
        <Sidebar/>
      <h1 className="about-title">About ClimaLung</h1>
      <p className="about-description">
        Air pollution is a major global health crisis, driven by harmful
        contaminants such as particulate matter (PM), carbon emissions, and
        toxic byproducts from industrial activities, fossil fuels, and vehicle
        exhausts. One of the most concerning pollutants is PM2.5, particulate
        matter with a diameter of 2.5 micrometers or smaller. These tiny particles
        can penetrate deep into the lungs and enter the bloodstream, posing
        significant health risks. Recognized as a carcinogen by the International
        Agency for Cancer Research, PM2.5 is a key contributor to lung cancer
        development in individuals exposed to high levels.
      </p>

      <p className="about-description">
        Pakistan, ranked as the second most polluted country in the world, faces
        severe challenges due to the combined effects of air pollution and its
        adverse health impacts. In cities like Lahore, the seasonal smog
        intensifies, worsening the crisis and increasing the prevalence of
        respiratory diseases and lung cancer. Given the urgency of the situation,
        there is an immediate need for solutions that can identify and correlate
        the harmful effects of air pollution on lung health. Moreover, the
        development of accurate and efficient tools for early lung cancer detection
        is essential to combat this growing threat.
      </p>

      <p className="about-description">
        ClimaLung leverages Artificial Intelligence (AI) to tackle this critical
        issue by integrating environmental data and healthcare technologies. Our
        AI-powered platform combines real-time air quality data with advanced
        diagnostic tools to explore the effects of climate change on lung health.
        By utilizing deep learning models, particularly Convolutional Neural
        Networks (CNNs), ClimaLung can detect early signs of lung cancer—offering
        a significant advancement over traditional diagnostic methods.
      </p>

      <p className="about-description">
        Beyond early detection, ClimaLung also identifies hazardous pollution
        levels, tracks air quality trends, and provides predictive insights on how
        these factors correlate with respiratory health. Our system bridges the gap
        between environmental monitoring and public health interventions,
        providing valuable data to healthcare providers, researchers, and
        policymakers.
      </p>

      <div className="about-cards">
        <div className="card">
          <div className="card-icon">
            <i className="fas fa-search"></i>
          </div>
          <h2 className="card-title">Research</h2>
          <p className="card-description">
            ClimaLung uses AI to analyze climate-related air pollution trends, such
            as PM2.5, PM10, and carbon emissions. It combines these insights with
            climate factors like temperature and humidity to understand their impact
            on human health.
          </p>
        </div>

        <div className="card">
          <div className="card-icon">
            <i className="fas fa-cogs"></i>
          </div>
          <h2 className="card-title">Strategy</h2>
          <p className="card-description">
            We develop predictive AI models that correlate environmental data with
            human respiratory health, aiming to provide early diagnosis and assist
            healthcare professionals in formulating intervention strategies.
          </p>
        </div>

        <div className="card">
          <div className="card-icon">
            <i className="fas fa-bell"></i>
          </div>
          <h2 className="card-title">Design</h2>
          <p className="card-description">
            The ClimaLung platform also incorporates AI-driven CT scan interpretation
            to enhance the accuracy of lung cancer diagnoses and aid healthcare
            authorities in timely interventions.
          </p>
        </div>
      </div>

      <div className="team-section">
        <h2 className="team-title">Our Team</h2>
        <div className="team-members">
          <div className="team-member">
            <h3 className="team-member-name">Shahmir Yousaf</h3>
            <p className="team-member-role">Team Member</p>
          </div>
          <div className="team-member">
            <h3 className="team-member-name">Wania Tariq</h3>
            <p className="team-member-role">Team Member</p>
          </div>
          <div className="team-member">
            <h3 className="team-member-name">Jannat Nasir</h3>
            <p className="team-member-role">Team Member</p>
          </div>
        </div>

        <h2 className="supervisor-title">Project Supervisors</h2>
        <div className="supervisors">
          <div className="supervisor">
            <h3 className="supervisor-name">Ms. Mehroze Khan</h3>
            <p className="supervisor-role">Supervisor</p>
          </div>
          <div className="supervisor">
            <h3 className="supervisor-name">Dr. Saman Shahid</h3>
            <p className="supervisor-role">Co-Supervisor</p>
          </div>
          <div className="supervisor">
            <h3 className="supervisor-name">Dr. Farukh Shahid</h3>
            <p className="supervisor-role">Co-Supervisor</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
