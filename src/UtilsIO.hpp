/*
 * UtilsIO.hpp
 *
 *  Created on: Apr 10, 2021
 */

#pragma once

#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

namespace Plato
{

namespace io
{

/******************************************************************************//**
 * \fn inline void open_text_file
 *
 * \brief Open text file.
 *
 * \param [in]     aFileName filename
 * \param [in]     aPrint    boolean flag (true = open file, false = do not open)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
inline void open_text_file
(const std::string & aFileName,
 std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile.open(aFileName);
}
// function open_text_file

/******************************************************************************//**
 * \fn inline void close_text_file
 *
 * \brief Close text file.
 *
 * \param [in]     aPrint    boolean flag (true = close file, false = do not close)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
inline void close_text_file
(std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile.close();
}
// function close_text_file

/******************************************************************************//**
 * \fn inline void append_text_to_file
 *
 * \brief Append text to file.
 *
 * \param [in]     aMsg      text message to be appended to file
 * \param [in]     aPrint    boolean flag (true = print message to file, false = do not print message)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
inline void append_text_to_file
(const std::stringstream & aMsg,
 std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile << aMsg.str().c_str() << std::flush;
}
// function append_text_to_file

}
// namespace io

namespace filesystem
{

/******************************************************************************//**
 * \fn exist
 *
 * \brief Return true if path exist; else, return false
 * \param [in] aPath directory/file path
 * \return boolean (true or false)
**********************************************************************************/
bool exist(const std::string &aPath)
{
    struct stat tBuf;
    int tReturn = stat(aPath.c_str(), &tBuf);
    return (tReturn == 0 ? true : false);
}
// function exist

/******************************************************************************//**
 * \fn exist
 *
 * \brief Delete file/directory if it exist
 * \param [in] aPath directory/file path
**********************************************************************************/
void remove(const std::string &aPath)
{
    if(Plato::filesystem::exist(aPath))
    {
        auto tCommand = std::string("rm -rf ") + aPath;
        auto tOutput = std::system(tCommand.c_str());
        if(false){ std::cout << std::to_string(tOutput) << "\n";}
    }
}
// function remove

}
// namespace filesystem

}
// namespace Plato
