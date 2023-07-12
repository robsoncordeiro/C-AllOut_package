//---------------------------------------------------------------------------
// object.h - Implementation of the User Layer
//
// This file implements the 2 classes required by the SlimTree Library User
// Layer.
//
// TCity is the object which will be indexed by a metric tree. Each object 
// has a name and n features. TCity defines an interface to manipulate its 
// information and also implements the stObject interface.
//
// TCityDistanceEvaluator implements the stMetricEvaluator interface. It will
// mesure the distance between 2 TCity instances.
//
// With these classes, it is possible to define and instantiate any metric tree
// defined by the SlimTree Library.
//
// Author: Guilherme Domingos Faria Silva
//---------------------------------------------------------------------------
#ifndef objectH
#define objectH

#include <math.h>
#include <string>
#include <time.h>
#include <ostream>
#include <bitset>
#include <bits/stdc++.h>
using namespace std;

// Metric Tree includes
#include <arboretum/stUserLayerUtil.h>
#include <arboretum/stTypes.h>
#include <arboretum/stUtil.h>

//---------------------------------------------------------------------------
// Class TCity
//---------------------------------------------------------------------------
/**
* Each object has a name and n features.
*
* <P>In addition to data manipulation methods , this class implements the 
* stObject interface. This interface qualifies this object to be indexed
* by a metric tree implemented by GBDI SlimTree Library.
*
* <P>This interface requires no inheritance (because of the use of class
* templates in the Structure Layer) but requires the following methods:
*     - TCity() - A default constructor.
*     - Clone() - Creates a clone of this object.
*     - IsEqual() - Checks if this instance is equal to another.
*     - GetSerializedSize() - Gets the size of the serialized version of this object.
*     - Serialize() - Gets the serialzied version of this object.
*     - Unserialize() - Restores a serialzied object.
*
* <P>Since the array which contains the serialized version of the object must be
* created and destroyed by each object instance, this class will hold this array
* as a buffer of the serialized version of this instance. This buffer will be
* created only if required and will be invalidated every time the object changes
* its values.
*
* @version 1.0
* @author Guilherme Domingos Faria Silva
*/
class TCity{
   public:
      /**
      * Default constructor. It creates an object with no name and longitude and
      * latitude set to 0. This constructor is required by stObject interface.
      */
      TCity(){
         name = "";
         instance.clear();
         instanceLength = 0;

         // Invalidate Serialized buffer.
         Serialized = NULL;
      }//end TCity

      /**
      * Creates a new object.
      *
      * @param name The name of the object.
      * @param latitude Latitude.
      * @param longitude Longitude.
      */
      TCity(const string name, vector<double> instance, int instanceLength){
         this->name = name;
         this->instance = instance;
         this->instanceLength = instanceLength;

         // Invalidate Serialized buffer.
         Serialized = NULL;
      }//end TCity

      /**
      * Destroys this instance and releases all associated resources.
      */
      ~TCity(){

         // Does Serialized exist ?
         if (Serialized != NULL){
            // Yes! Dispose it!
            delete [] Serialized;
         }//end if
      }//end TCity

      /**
      * Gets features of the object.
      */
      vector<double> GetInstance(){
         return instance;
      }//end GetInstance

      /**
      * Gets the name of the object.
      */
      const string & GetName(){
         return name;
      }//end GetName

      // The following methods are required by the stObject interface.
      /**
      * Creates a perfect clone of this object. This method is required by
      * stObject interface.
      *
      * @return A new instance of TCity wich is a perfect clone of the original
      * instance.
      */
      TCity * Clone(){
         return new TCity(name, instance, instanceLength);
      }//end Clone

      /**
      * Checks to see if this object is equal to other. This method is required
      * by  stObject interface.
      *
      * @param obj Another instance of TCity.
      * @return True if they are equal or false otherwise.
      */
      bool IsEqual(TCity *obj){
         return (instance == obj->GetInstance());
      }//end IsEqual

      /**
      * Returns the size of the serialized version of this object in bytes.
      * This method is required  by  stObject interface.
      */
      stSize GetSerializedSize(){
         return (sizeof(double) * instanceLength + name.length());
      }//end GetSerializedSize

      /**
      * Returns the serialized version of this object.
      * This method is required  by  stObject interface.
      *
      * @warning If you don't know how to serialize an object, this method may
      * be a good example.
      */
      const stByte * Serialize();

      /**
      * Rebuilds a serialized object.
      * This method is required  by  stObject interface.
      *
      * @param data The serialized object.
      * @param datasize The size of the serialized object in bytes.
      * @warning If you don't know how to serialize an object, this methos may
      * be a good example.
      */
      void Unserialize (const stByte *data, stSize datasize);

   private:
      /**
      * ID of the term.
      */
      string name;

      /**
      * Instance values.
      */
      vector<double> instance;

      /**
      * Instance length.
      */
      int instanceLength;

      /**
      * Serialized version. If NULL, the serialized version is not created.
      */
      stByte * Serialized;
};//end TMapPoint

//---------------------------------------------------------------------------
// Class TCityDistanceEvaluator
//---------------------------------------------------------------------------
/**
* This class implements a metric evaluator for TCity instances. It calculates
* the distance between objects by performing an Euclidean distance between object
* coordinates
*
* <P>It implements the stMetricEvaluator interface. As stObject interface, the
* stMetricEvaluator interface requires no inheritance and defines 2 methods:
*     - GetDistance() - Calculates the distance between 2 objects.
*     - GetDistance2()  - Calculates the distance between 2 objects raised by 2.
*
* <P>Both methods are defined due to optmization reasons. Since Euclidean
* distance raised by 2 is easier to calculate, It will implement GetDistance2()
* and use it to calculate GetDistance() result.
*
* @version 1.0
* @author Guilherme Domingos Faria Silva
*/
class TCityDistanceEvaluator : public stMetricEvaluatorStatistics{
   public:
      /**
      * Returns the distance between 2 objects. This method is required by
      * stMetricEvaluator interface.
      *
      * @param obj1 Object 1.
      * @param obj2 Object 2.
      */
      stDistance GetDistance(TCity *obj1, TCity *obj2){
         return sqrt(GetDistance2(obj1, obj2));
      }//end GetDistance

      /**
      * Returns the distance between 2 objects raised by the power of 2.
      * This method is required by stMetricEvaluator interface.
      *
      * @param obj1 Object 1.
      * @param obj2 Object 2.
      */
      stDistance GetDistance2(TCity *obj1, TCity *obj2){
         double delta1, delta2;

         UpdateDistanceCount(); // Update Statistics

         vector<double> delta;
         vector<double> a = obj1->GetInstance();
         vector<double> b = obj2->GetInstance();

         transform(a.begin(), a.end(), b.begin(), back_inserter(delta), [&](double l, double r)
            {
                double diff = l - r;
                return diff * diff;
            });

         return accumulate(delta.begin(), delta.end(), 0.0);
      }//end GetDistance2

      stDistance GetDistanceToProjection(TCity *obj, vector<double> * projection) {
         double delta1, delta2;

         UpdateDistanceCount(); // Update Statistics

         vector<double> delta;
         vector<double> a = obj->GetInstance();

         transform(a.begin(), a.end(), projection->begin(), back_inserter(delta), [&](double l, double r)
            {
                double diff = l - r;
                return diff * diff;
            });

         return sqrt(accumulate(delta.begin(), delta.end(), 0.0));
      }
};//end TCityDistanceEvaluator

//---------------------------------------------------------------------------
// Output operator
//---------------------------------------------------------------------------
/**
* This operator will write a string representation of an object to an outputstream.
*/
ostream & operator << (ostream & out, TCity & city);

#endif //end myobjectH
