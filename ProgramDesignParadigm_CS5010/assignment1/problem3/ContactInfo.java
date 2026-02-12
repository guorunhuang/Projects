package assignment1.problem3;

/**
 * Represents the contact details of a tax filer.
 */
public class ContactInfo {
  private Name name;
  private String address;
  private String phoneNumber;
  private String emailAddress;

  /**
   * Constructor for ContactInfo.
   * @param name Name object
   * @param address physical address
   * @param phoneNumber phone number
   * @param emailAddress email address
   */
  public ContactInfo(Name name, String address, String phoneNumber, String emailAddress) {
    this.name = name;
    this.address = address;
    this.phoneNumber = phoneNumber;
    this.emailAddress = emailAddress;
  }

  public Name getName() { return this.name; }
  public String getAddress() { return this.address; }
  public String getPhoneNumber() { return this.phoneNumber; }
  public String getEmailAddress() { return this.emailAddress; }
}