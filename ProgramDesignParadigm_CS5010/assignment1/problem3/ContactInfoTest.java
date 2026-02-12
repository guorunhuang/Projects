package assignment1.problem3;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the ContactInfo class.
 * Verifies the composition with the Name class and storage of contact details.
 */
public class ContactInfoTest {
  private ContactInfo testContact;
  private Name testName;
  private String expectedAddress;
  private String expectedPhone;
  private String expectedEmail;

  @BeforeEach
  public void setUp() {
    this.testName = new Name("First", "Last");
    this.expectedAddress = "310 Terry, Seattle, WA";
    this.expectedPhone = "206-666-0123";
    this.expectedEmail = "first.last@example.com";

    this.testContact = new ContactInfo(
        this.testName,
        this.expectedAddress,
        this.expectedPhone,
        this.expectedEmail
    );
  }

  @Test
  public void testGetName() {
    assertEquals(this.testName, this.testContact.getName(),
        "Name object should be correctly stored and retrieved.");
  }

  @Test
  public void testGetAddress() {
    assertEquals(this.expectedAddress, this.testContact.getAddress());
  }

  @Test
  public void testGetPhoneNumber() {
    assertEquals(this.expectedPhone, this.testContact.getPhoneNumber());
  }

  @Test
  public void testGetEmailAddress() {
    assertEquals(this.expectedEmail, this.testContact.getEmailAddress());
  }
}