package assignment1.problem1;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Deposit class.
 */
public class DepositTest {
  private static final String VALID_ID = "123456789012";
  private static final String VALID_NAME = "First Last";

  @Test
  public void testValidDeposit() {
    int validAmount = 5000;
    Deposit deposit = new Deposit(validAmount, VALID_ID, VALID_NAME);
    assertEquals(validAmount, deposit.getAmount());
    assertEquals(VALID_ID, deposit.getRecipientId());
  }

  @Test
  public void testDepositTooSmall() {
    // if in Deposit func we examine the range then throw IllegalArgumentException
    int smallAmount = 999;
    assertThrows(IllegalArgumentException.class, () -> {
      new Deposit(smallAmount, VALID_ID, VALID_NAME);
    });
  }

  @Test
  public void testDepositTooLarge() {
    int largeAmount = 10001;
    assertThrows(IllegalArgumentException.class, () -> {
      new Deposit(largeAmount, VALID_ID, VALID_NAME);
    });
  }
}