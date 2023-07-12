//---------------------------------------------------------------------------
// object.cpp - Implementation of the User Layer
//
// In this file we have the implementation of TCity::Serialize(),
// TCity::Unserialize() and an output operator for TCity (which is not required
// by user layer).
//
// Author: Guilherme Domingos Faria Silva
//---------------------------------------------------------------------------
#pragma hdrstop
#include "object.h"
#pragma package(smart_init)

//---------------------------------------------------------------------------
// Class TCity
//---------------------------------------------------------------------------
/**
* Returns the serialized version of this object.
* This method is required  by  stObject interface.
* @warning If you don't know how to serialize an object, this method may
* be a good example.
*/
const stByte * TCity::Serialize()
{
    double * s;

    // Is there a seralized version ?
    if (Serialized == NULL)
    {
        // No! Lets build the serialized version.

        // The first thing we need to do is to allocate resources...
        //cout << GetSerializedSize() << endl;
        Serialized = new stByte[GetSerializedSize()];

        s = (double *) Serialized; // If you are not familiar with pointers, this
        // action may be tricky! Be careful!

        //cout << s << endl;

        int i = 0;
        for(vector<double>::iterator it = instance.begin(); it != instance.end(); it++) {
            s[i] = *it;
            i++;
        }

        //s = &term

        //cout << s << "LINE BREAK" << endl;
        //cout << name.c_str() << endl;
        //cout << termLength << endl;

        // Now, write the name after the 2 doubles...
        memcpy(Serialized + (sizeof(double) * instance.size()), name.c_str(), name.length());

        //cout << s << endl;
    }//end if

    return Serialized;
}//end TCity::Serialize

/**
* Rebuilds a serialized object.
* This method is required  by  stObject interface.
*
* @param data The serialized object.
* @param datasize The size of the serialized object in bytes.
* @warning If you don't know how to serialize an object, this methos may
* be a good example.
*/
void TCity::Unserialize(const stByte *data, stSize datasize)
{
    stCount numFeatures = (datasize - 6) / sizeof(double);

    double * s;
    //stSize strl;

    s = (double *) data;  // If you are not familiar with pointers, this
    // action may be tricky! Be careful!

    //cout << s << endl;

    vector<double> sig;

    for (int i = 0; i < numFeatures; i++) {
        sig.push_back(s[i]);
    }

    //cout << term << endl;
    // To read the name, we must discover its size first. Since it is the only
    // variable length field, we can get it back by subtract the fixed size
    // from the serialized size.
    //strl = datasize - (sizeof(double) * 24);

    //cout << datasize << endl;
    //cout << strl << endl;

    // Now we know the size, lets get it from the serialized version.
    name.assign((char *)(data + (sizeof(double) * numFeatures)), 6);
    instance = sig;
    instanceLength = numFeatures;

    // Since we have changed the object contents, we must invalidate the old
    // serialized version if it exists. In fact we, may copy the given serialized
    // version of tbe new object to the buffer but we don't want to spend memory.
    if (Serialized != NULL)
    {
        delete [] Serialized;
        Serialized = NULL;
    }//end if
}//end TCity::Unserialize

//---------------------------------------------------------------------------
// Output operator
//---------------------------------------------------------------------------
/**
* This operator will write a string representation of an object to an outputstream.
*/
ostream & operator << (ostream & out, TCity & object)
{
    out << "[InstanceID=" << object.GetName();

    vector<double> instance = object.GetInstance();

    for(vector<double>::iterator it = instance.begin(); it != instance.end(); it++) {
        out << ";Instance=" << *it;
    }
    return out;
}//end operator <<
