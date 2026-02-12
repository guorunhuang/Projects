package assignment1.problem3;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the Name class.
 * Ensures that first and last names are correctly stored and retrieved.
 */
public class NameTest {
  private Name testName;
  private String expectedFirstName;
  private String expectedLastName;

  @BeforeEach
  public void setUp() {
    this.expectedFirstName = "First";
    this.expectedLastName = "Last";
    this.testName = new Name(this.expectedFirstName, this.expectedLastName);
  }

  @Test
  public void testGetFirstName() {
    assertEquals(this.expectedFirstName, this.testName.getFirstName(),
        "First name should match the value provided in constructor.");
  }

  @Test
  public void testGetLastName() {
    assertEquals(this.expectedLastName, this.testName.getLastName(),
        "Last name should match the value provided in constructor.");
  }
}