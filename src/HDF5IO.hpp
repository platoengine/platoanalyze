/*
 * HDF5Helper.hpp
 *
 *  Created on: Jul 24, 2019
 *      Author: doble
 */

#ifndef SRC_PLATO_HDF5IO_HPP_
#define SRC_PLATO_HDF5IO_HPP_

// HD5 c-interface
#include "hdf5.h"

#include <string>
#include <cassert>
#include <vector>

namespace Plato
{


/*
 * returns the parallel path provided a base path name
 */
std::string
make_path_parallel( const std::string & aPath,
                    int  aParSize = 1,
                    int  aRank = 0)
{
    // test if running in parallel mode
    if ( aParSize > 1 )
    {
        // get file extesion
        auto tFileExt = aPath.substr(
                aPath.find_last_of("."),
                aPath.length() );

        // get base path
        auto tBasePath = aPath.substr(
                0,
                aPath.find_last_of(".") );

        // add proc number to path
        std::string aParallelPath = tBasePath + "_"
                +  std::to_string( aParSize ) + "."
                +  std::to_string( aRank )
        + tFileExt;
        return aParallelPath;
    }
    else
    {
        // do not modify path
        return aPath;
    }
}

//------------------------------------------------------------------------------

    /**
     * create a new HDF5 file
     */
    hid_t
    create_hdf5_file( const std::string & aPath )
    {
        assert( aPath.size() > 0);
        return H5Fcreate(
                make_path_parallel(aPath).c_str(),
                H5F_ACC_TRUNC,
                H5P_DEFAULT,
                H5P_DEFAULT);
    }

//------------------------------------------------------------------------------

    /**
     * open an existing hdf5 file
     */
    hid_t
    open_hdf5_file( const std::string & aPath )
    {
        assert( aPath.size() > 0);

        // create parallel path
        std::string tPath = make_path_parallel( aPath );

        // test if file exists
        std::ifstream tFile( tPath );

        // throw error if file does not exist
        assert( tFile );

        // close file
        tFile.close();

        // open file as HDF5 handler
        return H5Fopen(
                tPath.c_str(),
                H5F_ACC_RDWR,
                H5P_DEFAULT);
    }

//------------------------------------------------------------------------------

    /**
     * close an open hdf5 file
     */
    herr_t
    close_hdf5_file( hid_t aFileID )
    {
        return H5Fclose( aFileID );
    }

//------------------------------------------------------------------------------

    /**
     * test if a dataset exists
     */
    bool
    dataset_exists( hid_t aFileID, const std::string & aLabel )
    {
        hid_t tDataSet = 0;
        return H5Lexists( aFileID, aLabel.c_str(), tDataSet );
    }

//------------------------------------------------------------------------------
    /**
     *
     * \brief                 returns a HDF5 enum defining the
     *                        data type that is to be communicated.
     *
     * \param[in] aSample     primitive data type with arbitrary value
     *
     * see also https://support.hdfgroup.org/HDF5/doc/H5.user/Datatypes.html
     */
    template < typename T > hid_t
    get_hdf5_datatype( const T & aSample )
    {
        assert( false );
        return 0;
    }

//------------------------------------------------------------------------------

    template <> hid_t
    get_hdf5_datatype( const int & aSample )
    {
        return H5T_NATIVE_INT;
    }

//------------------------------------------------------------------------------

    template <> hid_t
    get_hdf5_datatype( const long int & aSample )
    {
        return H5T_NATIVE_LONG;
    }

//------------------------------------------------------------------------------

    template <> hid_t
    get_hdf5_datatype( const unsigned int & aSample )
    {
        return H5T_NATIVE_UINT;
    }

//------------------------------------------------------------------------------

    template <> hid_t
    get_hdf5_datatype( const long unsigned int & aSample )
    {
        return H5T_NATIVE_ULONG;
    }

//------------------------------------------------------------------------------

    template <> hid_t
    get_hdf5_datatype( const double & aSample )
    {
        return H5T_NATIVE_DOUBLE;
    }

//------------------------------------------------------------------------------

    template <> hid_t
    get_hdf5_datatype( const long double & aSample )
    {
        return H5T_NATIVE_LDOUBLE;
    }

//------------------------------------------------------------------------------

    template<> hid_t
    get_hdf5_datatype( const bool & aSample )
    {
        return H5T_NATIVE_HBOOL;
    }

//------------------------------------------------------------------------------

    /**
     * this function returns true of both the HDF5 datatype
     * and the passed datatype have the same size
     */
    template < typename T >
    bool
    test_size_of_datatype( const T aValue )
    {
        return H5Tget_size( get_hdf5_datatype( aValue  ) ) == sizeof( T );
    }

//------------------------------------------------------------------------------

    /**
     * unpacks a std::vector and stores it into the file
     * file must be open
     *
     * \param[ inout ] aFileID  handler to hdf5 file
     * \param[ in ]    aLabel   label of vector to save
     * \param[ in ]    aVector  vector that is to be stored
     * \param[ in ]    aStatus  error handler
     *
     * see also
     * https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/
     */
    template < typename T >
    void
    save_vector_to_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            const std::vector< T >   & aVector,
            herr_t              & aStatus )
    {
        // check datatype
        assert(  test_size_of_datatype( ( T ) 0 ) );

        // test if dataset exists
        if ( dataset_exists( aFileID, aLabel ) )
        {
            // create message
            std::string tMessage = "The dataset " + aLabel + " can not be created because it already exist.";

            //MORIS_ERROR( false, tMessage.c_str() );
        }

        // get dimensions from vector
        hsize_t tSize = aVector.size();

        // create data space
        hid_t  tDataSpace = H5Screate_simple( 1, &tSize, NULL);

        // select data type for vector to save
        hid_t tDataType = H5Tcopy( get_hdf5_datatype( ( T ) 0 ) );

        // set data type to little endian
        aStatus = H5Tset_order( tDataType, H5T_ORDER_LE );
        hid_t tDataSet = 0;

        // create new dataset
        tDataSet = H5Dcreate(
                aFileID,
                aLabel.c_str(),
                tDataType,
                tDataSpace,
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT );

        // test if vector is not empty
        if( tSize > 0 )
        {
            // write data into dataset
            aStatus = H5Dwrite(
                    tDataSet,
                    tDataType,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    aVector.data());
        }

        // close open hids
        H5Sclose( tDataSpace );
        H5Tclose( tDataType );
        H5Dclose( tDataSet );

        // check for error
        assert( aStatus == 0 );
    }

    /**
     * unpacks a Plato::ScalarVector and stores it into the file
     * file must be open
     *
     * \param[ inout ] aFileID  handler to hdf5 file
     * \param[ in ]    aLabel   label of vector to save
     * \param[ in ]    aVector  vector that is to be stored
     * \param[ in ]    aStatus  error handler
     *
     * see also
     * https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/
     */
    void
    save_scalar_vector_to_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            typename Plato::ScalarVector::HostMirror   aScalarVector,
            herr_t              & aStatus )
    {
        // check datatype
        assert(  test_size_of_datatype( ( Plato::Scalar ) 0 ) );

        // test if dataset exists
        if ( dataset_exists( aFileID, aLabel ) )
        {
            // create message
            std::string tMessage = "The dataset " + aLabel + " can not be created because it already exist.";

            //MORIS_ERROR( false, tMessage.c_str() );
        }

        // get dimensions from vector
        hsize_t tSize = aScalarVector.extent(0);

        // create data space
        hid_t  tDataSpace = H5Screate_simple( 1, &tSize, NULL);

        // select data type for vector to save
        hid_t tDataType = H5Tcopy( get_hdf5_datatype( ( Plato::Scalar ) 0 ) );

        // set data type to little endian
        aStatus = H5Tset_order( tDataType, H5T_ORDER_LE );
        hid_t tDataSet = 0;

        // create new dataset
        tDataSet = H5Dcreate( aFileID, aLabel.c_str(), tDataType, tDataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

        // test if vector is not empty
        if( tSize > 0 )
        {
            // write data into dataset
            aStatus = H5Dwrite(tDataSet, tDataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, aScalarVector.data());
        }

        // close open hids
        H5Sclose( tDataSpace );
        H5Tclose( tDataType );
        H5Dclose( tDataSet );

        // check for error
        assert( aStatus == 0 );
    }

//------------------------------------------------------------------------------

    /**
     * unpacks a std::vector and stores it into the file
     * file must be open
     *
     * \param[ inout ] aFileID  handler to hdf5 file
     * \param[ in ]    aLabel   label of vector to save
     * \param[ out ]    aVector   vector that is to be loaded
     * \param[ in ]    aStatus  error handler
     *
     * see also
     * https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/
     */
    template < typename T >
    void
    load_vector_from_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            std::vector< T >    & aVector,
            herr_t              & aStatus )
    {
        // check datatype
        assert(  test_size_of_datatype( ( T ) 0 ) );

        // test if dataset exists
        if ( ! dataset_exists( aFileID, aLabel ) )
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: The dataset " + aLabel + " can not be opened because it does not exist.";
            throw std::runtime_error(tErrorMessage.str().c_str());

        }

        // open the data set
        hid_t tDataSet = H5Dopen1( aFileID, aLabel.c_str() );

        // get the data type of the set
        hid_t tDataType = H5Dget_type( tDataSet );

        // make sure that datatype fits to type of vector
        if ( H5Tget_class( tDataType ) !=  H5Tget_class( get_hdf5_datatype( ( T ) 0 ) ) )
        {
            std::string tMessage = "ERROR in reading from file: vector "
                    + aLabel + " has the wrong datatype.\n";

            std::cout<<tMessage<<std::endl;

            assert( false );
        }

        // get handler to dataspace
        hid_t tDataSpace = H5Dget_space( tDataSet );

        // vector dimensions
        hsize_t tDims[ 1 ];

        // ask hdf for dimensions
        aStatus  = H5Sget_simple_extent_dims( tDataSpace, tDims, NULL);

        // allocate memory for output
        aVector.resize( tDims[0] );

        // test if vector is not empty
        if( tDims[ 0 ] > 0 )
        {
            // read data from file
            aStatus = H5Dread(
                    tDataSet,
                    tDataType,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    aVector.data() );
        }
        else if( aStatus == 2 )
        {
            // all good, reset status
            aStatus = 0;
        }
        // Close/release resources
        H5Tclose( tDataType );
        H5Dclose( tDataSet );
        H5Sclose( tDataSpace );

        // check for error
        assert( aStatus == 0 );
    }

    /**
     * unpacks a Plato::ScalarVector and stores it into the file
     * file must be open
     *
     * \param[ inout ] aFileID  handler to hdf5 file
     * \param[ in ]    aLabel   label of vector to save
     * \param[ out ]    aVector   vector that is to be loaded
     * \param[ in ]    aStatus  error handler
     *
     * see also
     * https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/
     */

    void
    load_scalar_vector_from_hdf5_file(
            hid_t                                    & aFileID,
            const std::string                        & aLabel,
            typename Plato::ScalarVector::HostMirror & aVector,
            herr_t                                   & aStatus )
    {
        // check datatype
        assert(  test_size_of_datatype( ( Plato::Scalar ) 0 ) );

        // test if dataset exists
        if ( ! dataset_exists( aFileID, aLabel ) )
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: The dataset " + aLabel + " can not be opened because it does not exist.";
            throw std::runtime_error(tErrorMessage.str().c_str());

        }

        // open the data set
        hid_t tDataSet = H5Dopen1( aFileID, aLabel.c_str() );

        // get the data type of the set
        hid_t tDataType = H5Dget_type( tDataSet );

        // make sure that datatype fits to type of vector
        if ( H5Tget_class( tDataType ) !=  H5Tget_class( get_hdf5_datatype( ( Plato::Scalar ) 0 ) ) )
        {
            std::string tMessage = "ERROR in reading from file: vector "
                    + aLabel + " has the wrong datatype.\n";

            std::cout<<tMessage<<std::endl;

            assert( false );
        }

        // get handler to dataspace
        hid_t tDataSpace = H5Dget_space( tDataSet );

        // vector dimensions
        hsize_t tDims[ 1 ];

        // ask hdf for dimensions
        aStatus  = H5Sget_simple_extent_dims( tDataSpace, tDims, NULL);

        // allocate memory for output
        assert(tDims[0] == aVector.extent(0));

        // test if vector is not empty
        if( tDims[ 0 ] > 0 )
        {
            // read data from file
            aStatus = H5Dread(
                    tDataSet,
                    tDataType,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    aVector.data() );
        }
        else if( aStatus == 2 )
        {
            // all good, reset status
            aStatus = 0;
        }
        // Close/release resources
        H5Tclose( tDataType );
        H5Dclose( tDataSet );
        H5Sclose( tDataSpace );

        // check for error
        assert( aStatus == 0 );
    }

//------------------------------------------------------------------------------

    /**
     * saves a scalar value to a file
     * file must be open
     *
     * \param[ inout ] aFileID  handler to hdf5 file
     * \param[ in ]    aLabel   label of vector to save
     * \param[ in ]    aValue   value that is to be stored
     * \param[ in ]    aStatus  error handler
     */
    template < typename T >
    void
    save_scalar_to_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            const T             & aValue,
            herr_t              & aStatus )
    {
        // test if dataset exists
        if ( dataset_exists( aFileID, aLabel ) )
        {
            // create message
            std::string tMessage = "The dataset " + aLabel + " can not be created because it already exist.";

            std::cerr<<tMessage<<std::endl;

            // throw error
            assert( false );
        }

        // check datatype
        assert(  test_size_of_datatype( ( T ) 0 ) );

        // select data type for vector to save
        hid_t tDataType = H5Tcopy( get_hdf5_datatype( ( T ) 0 ) );

        // set data type to little endian
        aStatus = H5Tset_order( tDataType, H5T_ORDER_LE );

        // vector dimensions
        hsize_t tDims[ 1 ] = { 1 };

        // create data space
        hid_t tDataSpace = H5Screate_simple( 1, tDims, NULL );

        // create new dataset
        hid_t tDataSet = H5Dcreate(
                aFileID,
                aLabel.c_str(),
                tDataType,
                tDataSpace,
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT );

        // write data into dataset
        aStatus = H5Dwrite(
                tDataSet,
                tDataType,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                &aValue );

        // close open hids
        H5Sclose( tDataSpace );
        H5Tclose( tDataType );
        H5Dclose( tDataSet );

        // check for error
        assert( aStatus == 0 );
    }

//------------------------------------------------------------------------------

    /**
     * loads a scalar value from a file
     * file must be open
     *
     * \param[ inout ] aFileID  handler to hdf5 file
     * \param[ in ]    aLabel   label of vector to save
     * \param[ out ]   aValue  vector that is to be loaded
     * \param[ in ]    aStatus  error handler
     *
     * see also
     * https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/
     */
    template < typename T >
    void
    load_scalar_from_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            T                   & aValue,
            herr_t              & aStatus )
    {
        // test if dataset exists
        if ( ! dataset_exists( aFileID, aLabel ) )
        {
            // create message
            std::string tMessage = "The dataset " + aLabel + " can not be opened because it does not exist.";

            std::cerr<<tMessage<<std::endl;

            // throw error
            assert( false );
        }

        // check datatype
        assert(  test_size_of_datatype( ( T ) 0 ) );

        // open the data set
        hid_t tDataSet = H5Dopen1( aFileID, aLabel.c_str() );

        // get the data type of the set
        hid_t tDataType = H5Dget_type( tDataSet );

        // get handler to dataspace
        hid_t tDataSpace = H5Dget_space( tDataSet );

        // read data from file
        aStatus = H5Dread(
                tDataSet,
                tDataType,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                &aValue );

        // Close/release resources
        H5Tclose( tDataType );
        H5Dclose( tDataSet );
        H5Sclose( tDataSpace );

        // check for error
        assert( aStatus == 0 );
    }

//------------------------------------------------------------------------------

    void
    save_string_to_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            const std::string   & aValue,
            herr_t              & aStatus )
    {
        // test if dataset exists
        if ( dataset_exists( aFileID, aLabel ) )
        {
            // create message
            std::string tMessage = "The dataset " + aLabel + " can not be created because it already exist.";

            std::cerr<<tMessage<<std::endl;

            // throw error
            assert( false );
        }

        // select data type for string
        hid_t tDataType = H5Tcopy( H5T_C_S1 );

        // set size of output type
        aStatus  = H5Tset_size( tDataType, aValue.length() );

        // vector dimensions
        hsize_t tDims[ 1 ] = { 1 };

        // create data space
        hid_t tDataSpace = H5Screate_simple( 1, tDims, NULL );

        // create new dataset
        hid_t tDataSet = H5Dcreate(
                aFileID,
                aLabel.c_str(),
                tDataType,
                tDataSpace,
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT );

        // write data into dataset
        aStatus = H5Dwrite(
                tDataSet,
                tDataType,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                aValue.c_str() );

        // close open hids
        H5Sclose( tDataSpace );
        H5Tclose( tDataType );
        H5Dclose( tDataSet );

        // check for error
        assert( aStatus == 0 );
    }

//------------------------------------------------------------------------------

    void
    load_string_from_hdf5_file(
            hid_t               & aFileID,
            const std::string   & aLabel,
            std::string         & aValue,
            herr_t              & aStatus )
    {
        // test if dataset exists
        if ( ! dataset_exists( aFileID, aLabel ) )
        {
            // create message
            std::string tMessage = "The dataset " + aLabel + " can not be opened because it does not exist.";


            std::cerr<<tMessage<<std::endl;
            // throw error
            assert( false );
        }

        // open the data set
        hid_t tDataSet = H5Dopen1( aFileID, aLabel.c_str() );

        // get handler to dataspace
        hid_t tDataSpace = H5Dget_space( tDataSet );

        // get the data type of the set
        hid_t tDataType = H5Dget_type( tDataSet );

        // get length of string
        hsize_t tSize = H5Dget_storage_size( tDataSet );

        // allocate buffer for string
        char * tBuffer = (char * ) malloc( tSize * sizeof( char ) );

        // load string from hdf5
        aStatus = H5Dread(
                tDataSet,
                tDataType,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                tBuffer );

        // create string from buffer
        aValue.assign( tBuffer, tSize );

        // delete buffer
        free( tBuffer );

        // close open hids
        H5Sclose( tDataSpace );
        H5Tclose( tDataType );
        H5Dclose( tDataSet );

        // check for error
        assert( aStatus == 0 );
    }

//------------------------------------------------------------------------------



}


#endif /* SRC_PLATO_HDF5IO_HPP_ */
