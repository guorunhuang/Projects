package assignment1.problem1;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FrequentFlyerTest {

  @BeforeEach
  void setUp() {
    // temporary Db
    FrequentFlyer.database = new IDatabase() {
      @Override
      public boolean isValidRecipient(String id, String name) {
        // if ID > 10 then return True
        return id != null && id.length() >= 10;
      }

      @Override
      public void depositToAccount(String id, int amount) {
        // deposit
      }
    };
  }

  @Test
  void testSuccessfulTransfer() {
    MilesBalance myMiles = new MilesBalance(5000, 1000, 500);
    FrequentFlyer flyer = new FrequentFlyer("123456789012", "Rose", "M", "Last", "rose@test.com", myMiles);

    Deposit validDeposit = new Deposit(2000, "987654321098", "First Last");

    // so that database.isValidRecipient will not fail
    flyer.transferMiles(validDeposit);

    assertEquals(3000, myMiles.getTotalMiles());
  }
}