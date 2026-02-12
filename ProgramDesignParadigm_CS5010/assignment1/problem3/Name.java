package assignment1.problem3;

/**
 * Represents a tax filer's first and last name.
 */
public class Name {
  private String firstName;
  private String lastName;

  /**
   * Constructor for Name.
   * @param firstName first name of the filer
   * @param lastName last name of the filer
   */
  public Name(String firstName, String lastName) {
    this.firstName = firstName;
    this.lastName = lastName;
  }

  public String getFirstName() { return this.firstName; }
  public String getLastName() { return this.lastName; }
}