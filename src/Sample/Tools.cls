Class Sample.Tools Extends %AI.Tool
{

/// Tool to add a person to the database
Method AddPerson(name As %String, age As %Integer){
    Set person = ##class(Sample.Person).%New()
    Set person.Name = name
    Set person.Age = age
    Do person.%Save()
}
/// Tool query database for people younger than a specified age
Query GetPeopleYoungerThan(age As %Integer) As %SQLQuery(
    ROWSPEC = "Name:%String,Age:%Integer") [ SqlProc ]
{
    SELECT Name, Age From Sample.Person Where Age < :age
}

}
