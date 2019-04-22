/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	std::random_device rd;
	std::default_random_engine gen(rd());
  
  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);  

  num_particles = 100;  // TODO: Set the number of particles
  weights.resize(num_particles);
  for (int i = 0; i < num_particles; i++){
	Particle p;
	p.x = dist_x(gen);
	p.y = dist_y(gen);
	p.theta = dist_theta(gen);
	p.weight = 1.0;
	weights[i] = 1.0;
	p.id = i;
  	particles.push_back(p);
  }
 is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
 std::default_random_engine gen;
  
  // Make distributions for adding noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
 
  for (auto& p : particles){
	if (fabs(yaw_rate) < 0.000001) {
      		// predict without Gaussian noise
      		p.x += velocity * delta_t * cos(p.theta);
      		p.y += velocity * delta_t * sin(p.theta);
	} else {	  
		p.x = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
		p.y = p.y + (velocity/yaw_rate)*(cos(p.theta)-cos(p.theta + yaw_rate*delta_t));
		p.theta = p.theta + delta_t * yaw_rate;
	}
	p.x += dist_x(gen);
	p.y += dist_y(gen);
	p.theta += dist_theta(gen);
  }
 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  int i =0;
  double std_0 = std_landmark[0];
  double std_1 = std_landmark[1];  
  for (auto& p : particles){
    double w = 1.0;
    for (unsigned int j = 0; j < observations.size(); j++){
	   LandmarkObs obs = observations[j];
	   double new_x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
	   double new_y = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;
	   double min_x = 99999;
	   double min_y = 99999;
	   double min_distance = 999999;
   	   for (auto const &lm: map_landmarks.landmark_list){
   		double dx = p.x - lm.x_f;
		double dy = p.y - lm.y_f;
		if (dx*dx + dy*dy <= sensor_range*sensor_range){
			//Find minimum distance observation
			if (sqrt(pow(new_x - lm.x_f, 2) + pow(new_y - lm.y_f, 2)) < min_distance){
				min_x = lm.x_f;
				min_y = lm.y_f;
				min_distance = sqrt(pow(new_x - lm.x_f, 2) + pow(new_y - lm.y_f, 2));
			}
		}	
   	   }
	double X = new_x - min_x;
	double Y = new_y - min_y;
        w *= ( 1/(2*M_PI*std_0*std_1)) * exp( -( X*X/(2*std_0*std_0) + Y*Y/(2*std_1*std_1) ) );
    }
    p.weight = w;
    weights[i] = w;
    i++;
  }
 
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles (num_particles);
  
  // Use discrete distribution to return particles by weight
  std::random_device rd;
  std::default_random_engine gen(rd());
  for (int i = 0; i < num_particles; ++i) {
	  std::discrete_distribution<int> index(weights.begin(), weights.end());
    	  new_particles[i] = particles[index(gen)];
    
  }
  
  // Replace old particles with the resampled particles
  particles = new_particles;

  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
